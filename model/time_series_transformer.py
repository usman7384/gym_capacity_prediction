import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.position_embedding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        position = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        # Get embedding for each position
        pe = self.position_embedding[position, :]
        return x + pe

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, num_layers=3, d_model=64, nhead=4, dim_feedforward=256, max_len=5000):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.linear_in = nn.Linear(num_features, d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.linear_out = nn.Linear(d_model, 1)  # Output layer to predict occupancy_percentage

    def forward(self, src):
        # src shape: (batch_size, sequence_length, num_features)
        src = self.linear_in(src) * math.sqrt(self.d_model)  # Adjust dimensions and scale
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # Transformer expects (sequence_length, batch_size, d_model)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  # Convert back to (batch_size, sequence_length, d_model)
        output = self.linear_out(output[:, -1, :])  # Use output of the last time step
        return output
