from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable
from .utils import create_ffn

class DeepSetInvariantModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, dropout: float, activation: str, device: str):
        super(DeepSetInvariantModel, self).__init__()
        self.device = device
        self.phi = create_ffn(input_dim, output_dim, hidden_dim, num_layers, dropout, activation) 

        # rho is done later
        # self.rho = create_ffn(input_dim, output_dim, hidden_dim, num_layers, dropout, activation)

    def forward(self, f_subgraphs: FloatTensor, subgraph_scopes: List[List[int]]) -> FloatTensor:
        # num_subgraphs x input_size --> num_subgraphs x hidden_size
        # compute the representation for each data point
        phi_subgraphs = self.phi.forward(f_subgraphs)

        # get a mol's representation by summing the reps of its subgraphs
        # num_mols x hidden_size
        phi_mols = torch.empty(len(subgraph_scopes), f_subgraphs.shape[1], device=self.device)
        for i, scope in enumerate(subgraph_scopes):
            phi_mols[i] = torch.sum(phi_subgraphs[scope], dim=0)

        # compute the output
        # out = self.rho.forward(phi_mols)
        
        return phi_mols

