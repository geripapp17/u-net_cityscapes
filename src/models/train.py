import io

import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision.transforms.functional import pil_to_tensor

from PIL import Image
import matplotlib.pyplot as plt


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:

    train_loss = 0
    with tqdm(total=len(dataloader), leave=False, desc="Train") as pbar:
        for xs, ys in dataloader:
            xs, ys = xs.to(device), ys.to(device)

            use_amp = scaler is not None
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                preds = model(xs)

                preds = preds.transpose(1, 2).transpose(2, 3).contiguous().view(-1, preds.shape[1])
                ys = ys.view(-1)
                loss = loss_fn(preds, ys)

            train_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            pbar.update(1)

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
        with tqdm(total=len(dataloader), leave=False, desc="Test") as pbar:
            for xs, ys in dataloader:
                xs, ys = xs.to(device), ys.to(device)

                preds = model(xs)
                preds = preds.transpose(1, 2).transpose(2, 3).contiguous().view(-1, preds.shape[1])
                ys = ys.view(-1)

                loss = loss_fn(preds, ys)
                test_loss += loss.item()

                pbar.update(1)

    return test_loss / len(dataloader)


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    writer=None,
    model_saver=None,
    use_amp: bool = True,
) -> None:

    x_vis, y_vis = next(iter(test_dataloader))
    x_vis = x_vis.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            pbar.set_description(f"Epoch {epoch}")

            train_loss = train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )

            test_loss = test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device,
            )

            if writer:
                # TODO: Add wandb writer
                writer.add_scalars(
                    main_tag="Loss",
                    tag_scalar_dict={"train": train_loss, "test": test_loss},
                    global_step=epoch,
                )

                writer.add_image(
                    tag="Prediction-Target",
                    img_tensor=get_image_for_tensorboard(model, x_vis, y_vis),
                    global_step=epoch,
                )

            else:
                print(f"Epoch: {epoch}\t|\tTrain Loss: {train_loss}\t|\tTest Loss: {test_loss}")

            if model_saver is not None:
                model_saver(
                    current_loss=test_loss,
                    epoch=epoch,
                    model=model,
                    optim=optimizer,
                    loss_fn=loss_fn,
                    scaler=scaler,
                )

            pbar.update(1)

    if writer:
        writer.close()


def get_image_for_tensorboard(model, x, y):

    model.eval()
    with torch.inference_mode():
        visualize_pred = model(x)

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    visualize_pred, _ = torch.max(visualize_pred[0], dim=0, keepdim=True)
    axes[0].imshow(visualize_pred.permute(1, 2, 0).cpu().numpy())
    axes[0].axis(False)

    axes[1].imshow(y[0].numpy())
    axes[1].axis(False)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    tensor = pil_to_tensor(Image.open(io.BytesIO(buf.getvalue())))

    plt.close()

    return tensor
