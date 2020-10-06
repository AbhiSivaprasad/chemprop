import torch.nn as nn
from chemprop.nn_utils import get_activation_function

def create_ffn(input_size: int, output_size:int, hidden_size: int, layers: int, dropout: float, activation: str):
    activation = get_activation_function(activation)
    dropout = nn.Dropout(dropout)
    if layers == 1:
        ffn = [
            dropout,
            nn.Linear(input_size, output_size)
        ]
    else:
        ffn = [
            dropout,
            nn.Linear(input_size, hidden_size)
        ]
        for _ in range(args.layers - 2):
            ffn.extend([
                activation,
                dropout,
                nn.Linear(hidden_size, hidden_size),
            ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(hidden_size, output_size),
        ])

    # Create FFN model
    return nn.Sequential(*ffn)


