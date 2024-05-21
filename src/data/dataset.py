from pathlib import Path
from typing import Tuple
import glob
from enum import Enum

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class DATASET_NAME(Enum):
    CITYSCAPES = "CITYSCAPES"


class SPLIT_TYPE(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CITYSCAPES(Dataset):
    """CITYSCAPES Dataset."""

    IGNORED_CLASSES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    IGNORED_IDX = 255
    VALID_CLASSES = [IGNORED_IDX, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    CLASS_NAMES = [
        "unlabelled",
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic_light",
        "traffic_sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]
    COLORS = [
        (0, 0, 0),
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ]

    CLASS_MAP = dict(zip(VALID_CLASSES, range(len(VALID_CLASSES))))
    COLOR_MAP = dict(zip(range(len(VALID_CLASSES)), COLORS))

    def __init__(self, root: Path, split: SPLIT_TYPE, transform=None) -> None:

        self.root = root
        self.split = split
        self.transform = transform

        input_paths = sorted(glob.glob(f"{root / 'leftImg8bit' / split.value}/*/*.png"))
        target_paths = sorted(glob.glob(f"{root / 'gtFine' / split.value}/*/*labelIds.png"))

        self.data = [(input_paths[i], target_paths[i]) for i in range(len(input_paths))]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        img = np.array(Image.open(fp=self.data[idx][0]))
        target = self._encode_target(np.array(Image.open(fp=self.data[idx][1])))

        if self.transform is not None:
            transformed = self.transform(image=img, mask=target)
            img, target = transformed["image"], transformed["mask"]

        return img, target.long()

    def __len__(self) -> int:
        return len(self.data)

    def _encode_target(self, target: np.ndarray) -> np.ndarray:

        for idx in CITYSCAPES.IGNORED_CLASSES:
            target[target == idx] = CITYSCAPES.IGNORED_IDX

        for idx in CITYSCAPES.VALID_CLASSES:
            target[target == idx] = CITYSCAPES.CLASS_MAP[idx]

        return target
