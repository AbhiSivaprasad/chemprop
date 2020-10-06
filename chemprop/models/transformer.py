from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable

from .utils import create_ffn


class TransformerModel(nn.Module):
    def __init__(self, d_model: int, num_encoder_layers: int, device: str):
        super(TransformerModel, self).__init__()
        self.device = device
        self.encoder =  nn.TransformerEncoder(d_model=d_model, 
                                              num_encoder_layers=num_encoder_layers)


    def forward(self, f_subgraphs: FloatTensor, subgraph_scopes: List[List[int]]) -> FloatTensor:
        # prepare input to transformer
        max_seq_length = max(len(scope) for scope in subgraph_scopes)
        input_encodings = torch.zeros(max_seq_length, len(subgraph_scopes), -1, device=self.device)
        for i, scope in enumerate(subgraph_scopes):
            input_encodings[:, i, :len(scope)] = f_subgraphs[scope]
        
        # apply transformer
        transformer_encodings = self.encoder(input_encodings)

        # sum subgraph embeddings for each molecule
        molecule_encodings = torch.sum(transformer_encodings, dim=0) 

        # batch_size x embedding
        return molecule_encodings
