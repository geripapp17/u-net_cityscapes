import torch
from torch import nn
from tqdm import tqdm


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:

    train_loss = 0
    for xs, ys in dataloader:
        xs, ys = xs.to(device), ys.to(device)

        preds = model(xs)
        loss = loss_fn(preds, ys)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)


def test_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:

    model.eval()
    with torch.inference_mode():
        test_loss = 0
        for xs, ys in dataloader:
            xs, ys = xs.to(device), ys.to(device)

            preds = model(xs)
            loss = loss_fn(preds, ys)
            test_loss += loss.item()

    return test_loss / len(dataloader)


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    writer: torch.utils.tensorboard.writer.SummaryWriter = None,
) -> None:

    for epoch in tqdm(range(epochs)):

        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        print(f"Epoch: {epoch}\t|\tTrain Loss: {train_loss}\t|\tTest Loss: {test_loss}")

        if writer:
            # TODO: Add wandb writer
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train": train_loss, "test": test_loss},
                global_step=epoch,
            )

            writer.add_graph(model=model, input_to_model=torch.randn(1, 3, 512, 512).to(device))

    if writer:
        writer.close()
