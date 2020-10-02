from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable


class DeepSetInvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module, device: str):
        super(DeepSetInvariantModel, self).__init__()
        self.phi = phi
        self.rho = rho
        self.device = device

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
        out = self.rho.forward(phi_mols)
        
        return out


class Phi(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(Phi, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.model(x)


class Rho(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super(Rho, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_size),
        )


    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.model(x)
