from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable


class TransformerModel(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 d_model: int, 
                 num_encoder_layers: int, 
                 num_heads: int, 
                 dim_feedforward: int, 
                 dropout: float, 
                 device: str):
        super(TransformerModel, self).__init__()
        self.device = device
        self.input_projection = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                        nhead=num_heads, 
                                                        dropout=dropout, 
                                                        dim_feedforward=dim_feedforward)
        self.encoder =  nn.TransformerEncoder(encoder_layer=self.encoder_layer, 
                                              num_layers=num_encoder_layers)

    def forward(self, input_encodings: FloatTensor) -> FloatTensor:
        # apply transformer
        transformer_inputs = self.input_projection(input_encodings)
        transformer_encodings = self.encoder(transformer_inputs)

        return transformer_encodings
