from typing import List

import torch
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel


def predict(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    model.eval()

    preds = []

    for batches in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        batch_size = sum(map(len, batches))
        print(batch_size)

        # Prepare batch
        data_list = [(batch.batch_graph(), batch.features(), batch.atom_descriptors()) 
                     for batch in batches]
        
        # Make predictions
        with torch.no_grad():
            batch_preds = model(*list(zip(*data_list)))
            print(len(batch_preds))
            batch_preds = batch_preds[:batch_size]

        batch_preds = batch_preds.data.cpu().numpy()
        print("num batch preds", len(batch_preds))

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
        print("num preds", len(preds))

    return preds
