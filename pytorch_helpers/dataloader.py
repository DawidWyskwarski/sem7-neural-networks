from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

import torch


def create_dataloader(X, y, batch_size=32, shuffle=True, generator=None) -> DataLoader:
    return DataLoader(
        _DataframeDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


class _DataframeDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
