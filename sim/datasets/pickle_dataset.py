from __future__ import annotations

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class PickleDataset(Dataset):
    def __init__(self, dataset: str, root: str, train: bool):
        with open(os.path.join(root, f"{dataset}.pickle"), "rb") as f:
            data = pickle.load(f)
            if train:
                self.x = data[0]
                self.y = data[1]
            else:
                self.x = data[2]
                self.y = data[3]

        assert len(self.x) == len(self.y), "x and y should have the same length"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Assuming your dataframe's last column is the target/label
        features = torch.tensor(self.x[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx])  # type: ignore
        return features, label


if __name__ == "__main__":
    dataset = PickleDataset(
        dataset="signals", root="/home/mole/HDnn-RRAM-Opt/dataset", train=True
    )
    print(dataset[0])
