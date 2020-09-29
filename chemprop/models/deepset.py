from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable

NetIO = Union[FloatTensor, Variable]


class DeepSetInvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super(DeepSetInvariantModel).__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x: NetIO) -> NetIO:
        # compute the representation for each data point
        x = self.phi.forward(x)

        # sum up the representations
        # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
        # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.
        x = torch.sum(x, dim=0, keepdim=True)

        # compute the output
        out = self.rho.forward(x)

        return out


class Phi(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(Phi).__init__()
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
        super(Rho).__init__()

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
