import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image

from pathlib import Path
from typing import Tuple, List


class DatasetVOC(Dataset):
    """PASCAL Visual Object Classes (VOC) Dataset."""

    def __init__(self, names: List[str], input_path: Path, target_path: Path, transform=None) -> None:

        self.names = names
        self.input_path = input_path
        self.target_path = target_path
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        img = Image.open(fp=self.input_path / self.names[idx])
        target = Image.open(fp=self.target_path / self.names[idx])

        if self.transform:
            sample = self.transform(img, target)

        return sample

    def __len__(self) -> int:
        return len(self.names)
