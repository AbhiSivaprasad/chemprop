from typing import List, Union
from rdkit import Chem
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs, featurizer: bool = False):
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

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(args, atom_fdim=self.atom_fdim, bond_fdim=self.bond_fdim)

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
        Computes feature vectors of the input by running the model except for the last layer.

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
        self.subgraph_model = MoleculeModel(args, featurizer=True)
        graph_args = deepcopy(args)
        graph_args.depth = 0
        self.graph_model = MoleculeModel(args, atom_fdim=args.hidden_size, bond_fdim=1)

    def forward(self,
                batch_mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        assert features_batch is None

        # Encode subgraphs
        subgraph_encodings = self.subgraph_model(batch_mol_graph)

        # Build molecules from fully connected sets of subgraphs
        batch_mol_graph.connect_subgraphs(subgraph_encodings)

        # Encode molecules from fully connected sets of subgraphs
        graph_encodings = self.graph_model(batch_mol_graph)

        return graph_encodings
