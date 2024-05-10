from typing import Tuple
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import DATASET_NAME, SPLIT_TYPE, CITYSCAPES


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
            root=root,
            split=SPLIT_TYPE.TRAIN,
            transform=transform_train,
        )

        test_dataset = CITYSCAPES(
            root=root,
            split=SPLIT_TYPE.VAL,
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

    return train_dataloader, test_dataloader, len(train_dataset.CLASS_NAMES)


if __name__ == "__main__":

    train_dataloader, test_dataloader = get_dataloaders(
        dataset=DATASET_NAME.CITYSCAPES,
        root=Path("/home/geri/projects/machine_learning/datasets/cityscapes"),
        batch_size=16,
        num_workers=1,
        transform_train=None,
        transform_test=None,
    )
