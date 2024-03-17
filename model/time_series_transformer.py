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
    def __init__(self, num_features, config):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = config.d_model
        self.linear_in = nn.Linear(num_features, config.d_model)
        self.pos_encoder = LearnablePositionalEncoding(config.d_model, config.max_len if hasattr(config, 'max_len') else 5000)
        encoder_layer = TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead, dim_feedforward=config.dim_feedforward, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=config.num_layers)
        self.linear_out = nn.Linear(config.d_model, config.forecast_horizon)  # Use config.forecast_horizon

    def forward(self, src):
        src = self.linear_in(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # Transformer expects (sequence_length, batch_size, d_model)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  # Convert back to (batch_size, sequence_length, d_model)
        output = self.linear_out(output[:, -1, :])  # Use output of the last time step
        return output

