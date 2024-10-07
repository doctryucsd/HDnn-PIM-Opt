from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from pytorch_metric_learning import distances, losses, miners, reducers
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from sim.utils import calculate_acc

from .hd import HD


class HDTrainer:
    def __init__(self, training_args: Dict[str, Any]) -> None:
        self.device: str = training_args["device"]

    @torch.no_grad()
    def test(
        self,
        model: HD,
        test_loader: DataLoader,
    ):
        accs: List[float] = []
        weights: List[int] = []
        for x_test, y_test in test_loader:
            x_test_device = x_test.to(self.device)
            pred: Tensor = model(x_test_device)
            if y_test is not None:
                y_test_device = y_test.to(self.device)
                acc: float = calculate_acc(pred, y_test_device)
                accs.append(acc)
                weights.append(len(y_test))
        avg_acc = float(np.average(accs, weights=weights))

        return avg_acc

    # def metric_training(self, hd: HD, train_loader: DataLoader) -> None:
    #     # settings
    #     hd.train()
    #     optimizer = torch.optim.Adam(hd.parameters(), lr=self.metric_lr)

    #     # pytorch metric learning
    #     distance = distances.CosineSimilarity()
    #     reducer = reducers.ThresholdReducer(low=0.0)
    #     loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    #     mining_func = miners.TripletMarginMiner(
    #     margin=0.2, distance=distance, type_of_triplets="semihard", )

    #     # training loop
    #     for epoch in tqdm(range(self.metric_epochs)):
    #         for batch_idx, (x_train, y_train) in enumerate(train_loader):
    #             x_train_device: Tensor = x_train.to(self.device)
    #             y_train_device: Tensor = y_train.to(self.device)
    #             optimizer.zero_grad()
    #             embeddings, _ = hd._forward(x_train_device, True)
    #             indices_tuple: Tensor = mining_func(embeddings, y_train_device)
    #             loss: Tensor = loss_func(embeddings, y_train_device, indices_tuple)
    #             loss.backward()
    #             optimizer.step()
    #             # if batch_idx % 100 == 0:
    #             #     print(
    #             #         "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
    #             #             epoch, batch_idx, loss, mining_func.num_triplets
    #             #         )
    #             #     )
