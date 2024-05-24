import torch


def pixel_accuracy(preds: torch.Tensor, target: torch.Tensor) -> float:

    segm = torch.argmax(preds, dim=1)
    correct = torch.sum(segm == target)

    return correct / torch.numel(target)
