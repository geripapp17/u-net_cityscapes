import torch
import numpy as np
from torchvision import transforms

from typing import Tuple


class Rescale:
    def __init__(self, output_size: int | Tuple[int, int]) -> None:
        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size

    def __call__(self, img: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        height, width = img.shape[:2]
        if isinstance(self.output_size, int):
            if height > width:
                new_height, new_width = self.output_size * height / width, self.output_size
            else:
                new_height, new_width = self.output_size, self.output_size * width / height
        else:
            new_height, new_width = self.output_size

        new_height, new_width = int(new_height), int(new_width)

        transform = transforms.Resize(size=(new_height, new_width))

        return transform(img), transform(target)


class RandomCrop:
    def __init__(self, output_size: int | Tuple[int, int]) -> None:
        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        height, width = img.shape[:2]
        new_height, new_width = self.output_size

        p_y = np.random.randint(0, height - new_height + 1)
        p_x = np.random.randint(0, width - new_height + 1)

        return img[p_y : p_y + new_height, p_x : p_x + new_width], target[p_y : p_y + new_height, p_x : p_x + new_width]


class RandomHorizontalFlip:
    def __init__(self, p: float) -> None:
        assert p >= 0 and p <= 1

        self.p = p

    def __call__(self, img: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.rand() >= 1 - self.p:
            img = np.fliplr(img)
            target = np.fliplr(target)

        return img, target


class ToTensor:
    def __call__(self, img: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:

        img = img.transpose(axes=(2, 0, 1))
        target = target.transpose(axes=(2, 0, 1))

        return torch.from_numpy(img), torch.from_numpy(target)
