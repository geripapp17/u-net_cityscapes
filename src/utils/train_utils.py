import torch
import torch.nn as nn


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: nn.Module,
    device: torch.device,
) -> float:
    loss_train = 0.0

    model.train()
    for xs, ys in dataloader:
        xs, ys = xs.to(device), ys.to(device)

        preds = model(xs)
        loss = loss_fn(preds, ys)
        loss_train += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_train /= len(dataloader)

    return loss_train


def test_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    loss_test = 0.0

    model.eval()
    with torch.inference_mode():
        for xs, ys in dataloader:
            xs, ys = xs.to(device), ys.to(device)

            preds = model(xs)
            loss = loss_fn(preds, ys)
            loss_test += loss.item()

        loss_test /= len(dataloader)

    return loss_test
