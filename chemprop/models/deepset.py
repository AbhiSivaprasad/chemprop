from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable

NetIO = Union[FloatTensor, Variable]


class DeepSetInvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super(DeepSetInvariantModel, self).__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x: NetIO, split_sizes: List[int]) -> NetIO:
        # num_subgraphs x input_size --> num_subgraphs x hidden_size
        # compute the representation for each data point
        x = self.phi.forward(x)
        # sum up the representations
        # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
        # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.
        group_tensors = x.split_with_sizes(split_sizes)
        phi_list = [torch.sum(gt, dim=0, keepdim=True) for gt in group_tensors] 
        print(phi_list[0])
        print(phi_list[1])
        phi_tensor = torch.cat(phi_list, dim=0)  # batch_size x hidden_size

        # compute the output
        out = self.rho.forward(phi_tensor)
        
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

    def forward(self, x: NetIO) -> NetIO:
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


    def forward(self, x: NetIO) -> NetIO:
        return self.model(x)
