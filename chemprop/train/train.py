import wandb
import torch
import torch.nn as nn
import logging

from typing import Callable
from tensorboardX import SummaryWriter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.models import MoleculeModel
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: MoleculeModel,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for recording output. Is the string "wandb" when wandb logging is sought.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None and logger != "wandb" else print
    
    model.train()
    loss_sum = iter_count = 0
    acc_sum = 0

    wandb.watch(model, loss, log="all", log_freq=args.wandb_gradient_log_frequency)
    
    for batches in tqdm(data_loader, total=len(data_loader), leave=False):
        # Prepare batch
        batch_size = sum(map(len, batches))

        masks_list = []
        targets_list = []
        data_list = []
        for batch in batches:
            mol_batch, features_batch, target_batch, atom_descriptors_batch = \
                batch.batch_graph(), batch.features(), batch.targets(), batch.atom_descriptors()
            mask = masks_list.append(torch.Tensor([[x is not None for x in tb] for tb in target_batch]))
            targets_list.append(torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]))
            data_list.append((mol_batch, features_batch, atom_descriptors_batch))

        mask = torch.cat(masks_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        
        # Run model
        model.zero_grad()
        
        preds = model(*list(zip(*data_list))) #mol_batch, features_batch, atom_descriptors_batch)
        preds = preds[:batch_size]
        pred_classes = (preds > 0).int()

        # Move tensors to correct device
        mask = mask.to(preds.device)
        targets = targets.to(preds.device)
        class_weights = torch.ones(targets.shape, device=preds.device)

        acc = None
        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
            acc = (pred_classes == targets) * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask
            acc = (pred_classes == targets) * mask
        loss = loss.sum() / mask.sum()
        acc = acc.sum() / mask.sum()

        loss_sum += loss.item()
        acc_sum += acc
        iter_count += 1
        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += batch_size

        # Logging
        lrs = scheduler.get_lr()
        pnorm = compute_pnorm(model)
        gnorm = compute_gnorm(model)
        loss_avg = loss_sum / iter_count
        acc_avg = acc_sum / iter_count
        acc_sum = loss_sum = iter_count = 0
        lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))

        # Log to wandb after every batch
        if args.wandb:
            log = {"train_loss": loss.item(), "avg_train_loss": loss_avg, "param norm": pnorm, "gradient_norm": gnorm}
            for i, lr in enumerate(lrs):
                log[f"learning_rate_{i}"] = lr
            wandb.log(log)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_acc', acc_avg, n_iter)
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
