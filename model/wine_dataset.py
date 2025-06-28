from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import pandas as pd
import torch


class WineDataset(Dataset):
    def __init__(self, X_path, y_path):

        self.X = pd.read_parquet(X_path).astype(np.float32).values
        self.y = pd.read_parquet(y_path).astype(np.float32).squeeze().values

        assert len(self.X) == len(self.y), "Niezgodna liczba próbek X i y"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx]), torch.tensor(self.y[idx]))


class WineQualityMLP(nn.Module):
    def __init__(
        self, input_size: int, h1: int = 128, h2: int = 64, dropout: float = 0.2
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
