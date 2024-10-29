from __future__ import annotations

import os
from logging import Logger
from typing import Any, Dict, List

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .pickle_dataset import PickleDataset
from .ucihar import UCIHAR


def split_train_val_sets(
    split, train_set_all: datasets.VisionDataset | UCIHAR | PickleDataset
):
    all_len = len(train_set_all)
    train_len = int(all_len * split)

    train_set = Subset(train_set_all, range(train_len))
    val_set = Subset(train_set_all, range(train_len, all_len))
    return train_set, val_set


def dataset_factory(dataset_name: str, root: str, transform, train: bool):
    if dataset_name == "mnist":
        return datasets.MNIST(
            root=root, transform=transform, train=train, download=True
        )
    elif dataset_name == "cifar10":
        return datasets.CIFAR10(
            root=root, transform=transform, train=train, download=True
        )
    elif dataset_name == "fashion_mnist":
        return datasets.FashionMNIST(
            root=root, transform=transform, train=train, download=True
        )
    elif dataset_name == "ucihar":
        return UCIHAR(root, train)
    elif dataset_name == "pampa":
        return PickleDataset("pampa", root, train)
    elif dataset_name == "face":
        return PickleDataset("face", root, train)
    elif dataset_name == "isolet":
        return PickleDataset("isolet", root, train)
    else:
        raise NotImplementedError("Unknown dataset name: ", dataset_name)


def transform_factory(dataset: str):
    transform_list: List = [transforms.ToTensor()]
    # if dataset in ["cifar10"]:
    #     transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    # elif dataset in ["mnist", "fashion_mnist"]:
    #     transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    return transforms.Compose(transform_list)


def get_dataset(
    dataset_name: str,
    cwd: str,
    data_args: Dict[str, Any],
    eval: bool,
    logger: Logger | None,
):
    split: float = data_args["train_ratio"] if not eval else 1.0

    assert 0 < split <= 1, "train_ratio must be in (0, 1]"

    # dataset root
    root: str = os.path.join(cwd, "dataset")

    # transform
    transform = transform_factory(dataset_name)

    # load dataset
    train_set_all = dataset_factory(dataset_name, root, transform, True)
    test_set = dataset_factory(dataset_name, root, transform, False)

    # split train, val, test sets
    train_set, val_set = split_train_val_sets(split, train_set_all)

    if logger is not None:
        logger.info(f"image size: {train_set[0][0].shape}")

    return test_set, train_set, val_set


def get_dataloader(
    data_args: Dict[str, Any],
    test_set: Dataset,
    val_set: Dataset | None,
    train_set: Dataset,
):
    train_batch_size: int = data_args["train_batch_size"]
    test_batch_size: int = data_args["test_batch_size"]
    num_workers: int = data_args["num_workers"]

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
    )
    if val_set is not None:
        val_loader = DataLoader(
            val_set, batch_size=train_batch_size, num_workers=num_workers
        )
    else:
        val_loader = None

    test_loader = DataLoader(
        test_set, batch_size=test_batch_size, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def load_dataloader(
    dataset_name: str,
    cwd: str,
    data_args: Dict[str, Any],
    eval: bool,
    logger: Logger | None = None,
):
    # args
    test_set, train_set, val_set = get_dataset(
        dataset_name, cwd, data_args, eval, logger
    )

    train_loader, val_loader, test_loader = get_dataloader(
        data_args, test_set, val_set, train_set
    )

    return train_loader, val_loader, test_loader
