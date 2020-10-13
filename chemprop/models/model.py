from typing import List, Union
from rdkit import Chem
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

from .mpn import MPN
from .deepset import DeepSetInvariantModel
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights
from .transformer import TransformerModel
from .utils import create_ffn

class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs, atom_fdim = None, bond_fdim = None, featurizer: bool = False):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e., outputting the
                           learned features from the last layer prior to prediction rather than
                           outputting the actual property predictions.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args, atom_fdim, bond_fdim)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs, atom_fdim = None, bond_fdim = None) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(args, atom_fdim, bond_fdim)        

    def create_ffn(self, args: TrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size

        if args.atom_descriptors == 'descriptor':
            first_linear_dim += args.atom_descriptors_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def featurize(self,
                  batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                  features_batch: List[np.ndarray] = None,
                  atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Computes feature vectors of the input by running the model except for the last layer.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :return: The feature vectors computed by the :class:`MoleculeModel`.
        """
        return self.ffn[:-1](self.encoder(batch, features_batch, atom_descriptors_batch))

    def forward(self,
                batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :return: The output of the :class:`MoleculeModel`, which is either property predictions
                 or molecule features if :code:`self.featurizer=True`.
        """
        if self.featurizer:
            return self.featurize(batch, features_batch, atom_descriptors_batch)

        output = self.ffn(self.encoder(batch, features_batch, atom_descriptors_batch))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output


class KGModel(nn.Module):
    def __init__(self, args: TrainArgs):
        super(KGModel, self).__init__()
        self.device = args.device
        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'

        self.subgraph_model = MPN(args)

        # compute output size
        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes
            self.num_classes = args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        subgraph_encoding_dim = args.hidden_size

        # subgraph embeddings --> molecule embedding
        if args.kg_molecule_model == 'deepset':
            self.molecule_encoder = DeepSetInvariantModel(input_dim=subgraph_encoding_dim, 
                                                          output_dim=subgraph_encoding_dim, 
                                                          hidden_dim=subgraph_encoding_dim, 
                                                          num_layers=args.deepset_num_layers, 
                                                          dropout=args.kg_molecule_model_dropout, 
                                                          activation=args.activation,
                                                          device=self.device)

            self.molecule_embed_dim = subgraph_encoding_dim

        elif args.kg_molecule_model == 'transformer': 
            self.molecule_encoder = TransformerModel(input_dim=subgraph_encoding_dim,
                                                     d_model=args.transformer_feature_dim, 
                                                     num_encoder_layers=args.transformer_num_encoder_layers,
                                                     num_heads=args.transformer_num_heads,
                                                     dropout=args.transformer_dropout,
                                                     device=self.device)
        
            self.molecule_embed_dim = args.transformer_feature_dim
        else:
            self.molecule_encoder = None
            self.molecule_embed_dim = subgraph_encoding_dim

        self.ffn = create_ffn(self.molecule_embed_dim, 
                              self.output_size, 
                              args.ffn_hidden_size, 
                              args.ffn_num_layers, 
                              args.dropout, 
                              args.activation)

    def forward(self,
                batch_mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        #assert features_batch is None
        #assert atom_descriptors_batch is None

        # Encode subgraphs
        device = self.ffn[1].weight.device
        gpu_id = int(str(device)[-1:])
        if gpu_id >= len(batch_mol_graph):
            return torch.tensor([[0]], device=device)

        batch_mol_graph = batch_mol_graph[gpu_id]
        subgraph_encodings = self.subgraph_model(batch_mol_graph)
        print(f"## of subgraph encodings: {subgraph_encodings.shape[0]}")
        
        # organize subgraph encodings by molecule
        max_seq_length = max(len(scope) for scope in batch_mol_graph.subgraph_scope)
        input_encodings = torch.zeros(
            max_seq_length, 
            len(batch_mol_graph.subgraph_scope), 
            subgraph_encodings.shape[1], 
            device=self.ffn[1].weight.device
        )
        
        # num_subgraphs, molecule, embedding
        for i, scope in enumerate(batch_mol_graph.subgraph_scope):
            input_encodings[:len(scope), i, :] = subgraph_encodings[scope]

        # pass just the embeddings and scopes of molecules
        molecule_subgraph_encodings = self.molecule_encoder(input_encodings) if self.molecule_encoder else input_encodings

        # aggregate subgraph encodings of each molecule to get moleucle encodings
        molecule_encodings = torch.mean(molecule_subgraph_encodings, dim=0)
       
        output = self.ffn(molecule_encodings)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)

        if self.multiclass and not self.training:
            # batch size x num targets x num classes per target
            output = output.reshape((output.size(0), -1, args.num_classes))  
            if not self.training:
                # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
                output = self.multiclass_softmax(output) 
        #print(output.shape)
        return output

