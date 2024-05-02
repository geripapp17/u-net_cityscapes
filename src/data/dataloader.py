from torch.utils.data import DataLoader
from typing import Tuple, List
from pathlib import Path

from .dataset import DATASET_NAME, CITYSCAPES


def get_dataloaders(
    dataset: DATASET_NAME,
    root: Path,
    batch_size: int,
    num_workers: int,
    transform_train=None,
    transform_test=None,
) -> Tuple[DataLoader, DataLoader]:

    if dataset == DATASET_NAME.CITYSCAPES:
        train_dataset = CITYSCAPES(
            input_root=root / "leftImg8bit/train",
            target_root=root / "gtFine/train",
            transform=transform_train,
        )

        test_dataset = CITYSCAPES(
            input_root=root / "leftImg8bit/test",
            target_root=root / "gtFine/test",
            transform=transform_test,
        )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_dataloader, test_dataloader
