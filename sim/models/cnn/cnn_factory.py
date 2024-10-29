from __future__ import annotations

import torch
import torch.optim as optim
from torch import Tensor, nn
from torch.utils.data import DataLoader

from neurosim.Inference_pytorch.modules import QConv2d

from .cnn import CNN
from .cnn_train import CNNTrain


class CNNFactory:
    def __init__(
        self,
        in_channels_1: int,
        out_channels_1: int,
        kernel_size_1: int,
        stride_1: int,
        padding_1: int,
        dilation_1: int,
        out_channels_2: int,
        kernel_size_2: int,
        stride_2: int,
        padding_2,
        dilation_2: int,
        device: str | int,
    ) -> None:
        cnn1 = nn.Conv2d(
            in_channels_1,
            out_channels_1,
            kernel_size_1,
            stride_1,
            padding_1,
            dilation_1,
        )
        cnn2 = nn.Conv2d(
            out_channels_1,
            out_channels_2,
            kernel_size_2,
            stride_2,
            padding_2,
            dilation_2,
        )
        self.cnn = CNN(cnn1, cnn2).to(device)
        self.device = device

    def get_output_dim(self, image: Tensor) -> int:
        self.cnn.eval()

        assert (
            image.shape[0] == 1
        ), f"Batch size must be 1, got image.shape: {image.shape}"
        with torch.no_grad():
            output: Tensor = self.cnn(image)
        assert (
            len(output.shape) == 2
        ), f"Output must be 2D, got output.shape: {output.shape}"

        self.output_dim: int = output.shape[1]
        return self.output_dim

    def train(
        self,
        train_loader: DataLoader,
        num_classes: int,
        inner_dim: int,
        epochs: int,
        lr: float,
        verbose: bool = False,
    ) -> None:
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.cnn.parameters(), lr=lr)

        cnn_train = CNNTrain(self.cnn, self.output_dim, num_classes, inner_dim).to(
            self.device
        )

        # Training loop
        for epoch in range(epochs):
            cnn_train.train()
            running_loss: float = 0.0
            correct_predictions: int = 0
            total_predictions: int = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs: Tensor = cnn_train(inputs)
                loss: Tensor = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # record
                running_loss += loss.item()
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

            if verbose:
                acc = correct_predictions / total_predictions
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Acc: {acc:.4f}"
                )

    def clone_param(self, cnn: CNN) -> None:
        self.cnn.cnn1.weight.data = cnn.cnn1.weight.data.clone()
        self.cnn.cnn2.weight.data = cnn.cnn2.weight.data.clone()

    def create(self) -> CNN:
        return self.cnn

    def create_neurosim(self) -> CNN:
        cnn1_neurosim = QConv2d(self.cnn.cnn1.in_channels, self.cnn.cnn1.out_channels, self.cnn.cnn1.kernel_size, self.cnn.cnn1.stride, self.cnn.cnn1.padding, self.cnn.cnn1.dilation, name="CNN1")  # type: ignore
        cnn1_neurosim.weight.data = self.cnn.cnn1.weight.data.clone()

        cnn2_neurosim = QConv2d(self.cnn.cnn2.in_channels, self.cnn.cnn2.out_channels, self.cnn.cnn2.kernel_size, self.cnn.cnn2.stride, self.cnn.cnn2.padding, self.cnn.cnn2.dilation, name="CNN2")  # type: ignore
        cnn2_neurosim.weight.data = self.cnn.cnn2.weight.data.clone()
        return CNN(cnn1_neurosim, cnn2_neurosim).to(self.device)
