import torch


def pixel_accuracy(preds: torch.Tensor, ys: torch.Tensor) -> float:

    segm = torch.argmax(preds, dim=1)
    correct = torch.sum(segm == ys)

    return correct / torch.numel(ys)
