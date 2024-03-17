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

        # Ensure 'timestamp' is a datetime type and sort by it
        self.data_frame['timestamp'] = pd.to_datetime(self.data_frame['timestamp'])
        self.data_frame.sort_values('timestamp', inplace=True)
        
        self.features = self.data_frame.drop(['occupancy_percentage', 'timestamp'], axis=1)
        self.labels = self.data_frame['occupancy_percentage']

        # Generate valid indices ensuring continuity
        self.valid_indices = self._find_valid_sequence_indices()

    def _find_valid_sequence_indices(self):
        valid_indices = []
        timestamps = self.data_frame['timestamp']
        max_idx = len(self.data_frame) - self.sequence_length - self.forecast_horizon + 1

        for idx in range(max_idx):
            # Calculate the time difference between the first and the last timestamp in the sequence
            time_diff = (timestamps.iloc[idx + self.sequence_length - 1] - timestamps.iloc[idx]).total_seconds() / 3600
            
            # Check if the time difference is exactly what we expect (24 hours for sequence_length of 96 with 15 min intervals)
            if time_diff == (self.sequence_length - 1) * 15 / 60:
                valid_indices.append(idx)
            else:
                print(f"Invalid sequence at index {idx}, time difference: {time_diff} hours")
        
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Use valid index to ensure continuity
        actual_idx = self.valid_indices[idx]

        start = actual_idx
        end = actual_idx + self.sequence_length
        features_sequence = self.features.iloc[start:end].values
        target_sequence = self.labels.iloc[end:end+self.forecast_horizon].values

        # Convert boolean to integers and ensure all data is float32 for compatibility
        features_sequence = features_sequence.astype(np.float32)

        # Convert to tensor
        features_sequence = torch.from_numpy(features_sequence).float()
        target_sequence = torch.from_numpy(target_sequence).float()

        if self.transform:
            features_sequence = self.transform(features_sequence)

        return features_sequence, target_sequence
