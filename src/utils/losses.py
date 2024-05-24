import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()

        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> float:

        return 1 - self._multi_class_dice_coeff(logits=logits, target=target)

    def _multi_class_dice_coeff(self, logits: torch.Tensor, target: torch.Tensor) -> float:

        num_classes = logits.shape[1]
        target_single_channel = torch.unsqueeze(target, dim=1)

        true_one_hot = torch.eye(num_classes, device=target.device)[target]
        true_one_hot = true_one_hot.permute(0, 3, 1, 2).float()

        probs = F.softmax(logits, dim=1)

        true_one_hot = true_one_hot.type(logits.type())
        dims = (0,) + tuple(range(2, target_single_channel.ndimension()))

        intersection = torch.sum(probs * true_one_hot, dim=dims)
        cardinality = torch.sum(probs + true_one_hot, dim=dims)

        return (2.0 * intersection / (cardinality + self.eps)).mean()
