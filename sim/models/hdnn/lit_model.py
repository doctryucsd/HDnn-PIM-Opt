from __future__ import annotations

from typing import List

import numpy as np
from pytorch_lightning import LightningModule
from torch import nn, optim

from sim.utils import calculate_acc

from .cnn import CNN
from .pretrain import PreTrain


class LitModel(LightningModule):
    def __init__(self, num_classes: int, input_channel: int, cnn_output_dim: int):
        super().__init__()
        cnn = CNN(input_channel)
        self.model = PreTrain(cnn, num_classes, cnn_output_dim)
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_accs: List[float] = []
        self.val_weights: List[int] = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.loss_fn(z, y)

        self.log("train_loss", loss)
        # print(f"train_loss: {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        pred = z.argmax(dim=1)
        acc = calculate_acc(pred, y)

        # store validation loss
        self.val_accs.append(acc)
        self.val_weights.append(len(y))
        assert len(y) == len(x), f"len(y)={len(y)}, len(x)={len(x)}"

    def on_validation_epoch_end(self):
        assert len(self.val_accs) == len(
            self.val_weights
        ), f"len(self.val_accs)={len(self.val_accs)}, len(self.val_weights)={len(self.val_weights)}"

        # Aggregate validation acc
        avg_val_acc = float(np.average(self.val_accs, weights=self.val_weights))
        self.log("val_acc", avg_val_acc, on_epoch=True, prog_bar=True, logger=True)

        # Reset validation loss
        self.val_accs = []
        self.val_weights = []

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
