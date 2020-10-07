from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable


class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_encoder_layers: int, num_heads: int, dropout: float, device: str):
        super(TransformerModel, self).__init__()
        self.device = device
        self.input_projection = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.encoder =  nn.TransformerEncoder(encoder_layer=self.encoder_layer, 
                                              num_layers=num_encoder_layers)


    def forward(self, input_encodings: FloatTensor, subgraph_scopes: List[List[int]]) -> FloatTensor:
        # prepare input to transformer
       
        # apply transformer
        transformer_inputs = self.input_projection(input_encodings)
        transformer_encodings = self.encoder(transformer_inputs)

        # sum subgraph embeddings for each molecule
        molecule_encodings = torch.sum(transformer_encodings, dim=0) 

        # batch_size x embedding
        return molecule_encodings
