import torch
from torch.utils.data import Dataset
import pandas as pd

class GymOccupancyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with preprocessed data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

        # Features and labels are explicitly defined
        self.features = self.data_frame.drop('occupancy_percentage', axis=1).values
        self.labels = self.data_frame['occupancy_percentage'].values

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        if self.transform:
            features = self.transform(features)

        return features, label
