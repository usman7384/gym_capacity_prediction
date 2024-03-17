import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class GymOccupancyDataset(Dataset):
    def __init__(self, csv_file, sequence_length=96, forecast_horizon=8, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        # Assuming 'timestamp' is not needed as a feature for model input
        self.features = self.data_frame.drop(['occupancy_percentage', 'timestamp'], axis=1)
        self.labels = self.data_frame['occupancy_percentage']

    def __len__(self):
        # Adjust length to account for the sequence_length and forecast_horizon
        return len(self.data_frame) - self.sequence_length - self.forecast_horizon + 1

    def __getitem__(self, idx):
        start = idx
        end = idx + self.sequence_length
        features_sequence = self.features.iloc[start:end].values
        target_sequence = self.labels.iloc[end:end+self.forecast_horizon].values

        # Convert to tensor
        features_sequence = torch.from_numpy(features_sequence).float()
        target_sequence = torch.from_numpy(target_sequence).float()

        if self.transform:
            features_sequence = self.transform(features_sequence)

        return features_sequence, target_sequence