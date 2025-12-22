from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class UCIHAR(Dataset):
    def __init__(self, root: str, train: bool):
        if train:
            file_path = os.path.join(root, "ucihar/train.csv")
        else:
            file_path = os.path.join(root, "ucihar/test.csv")

        self.data = pd.read_csv(file_path, header=None)
        self.class_dict = {
            "WALKING": 0,
            "WALKING_UPSTAIRS": 1,
            "WALKING_DOWNSTAIRS": 2,
            "SITTING": 3,
            "STANDING": 4,
            "LAYING": 5,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming your dataframe's last column is the target/label
        features = torch.tensor(self.data.iloc[idx, :-2].values.astype(np.float32))
        raw_label = self.data.iloc[idx, -1]
        if isinstance(raw_label, str):
            label = torch.tensor(self.class_dict[raw_label])  # type: ignore
        else:
            label_value = int(raw_label)
            if label_value >= 1:
                label_value -= 1
            label = torch.tensor(label_value)
        return features, label


if __name__ == "__main__":
    dataset = UCIHAR(root="/home/mole/HDnn-RRAM-Opt/dataset", train=True)
    print(dataset[0])
    print(len(dataset[0][0]))
    labels = [float(dataset[i][1]) for i in range(len(dataset))]
    label_set = set(labels)
    max_label = max(labels)
    print(label_set)
    print(max_label)
