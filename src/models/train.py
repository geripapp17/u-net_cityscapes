import io
from typing import Dict

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
) -> Dict[str, float]:

    metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
    }

    with tqdm(total=len(dataloader), leave=False, desc="Train") as pbar:
        for xs, ys in dataloader:
            xs, ys = xs.to(device), ys.to(device)

            with torch.autocast(device_type=device, enabled=scaler.is_enabled(), dtype=torch.float16):
                preds = model(xs)
                loss = loss_fn(preds, ys)

            metrics["loss"] += loss.item()
            metrics["accuracy"] += pixel_accuracy(preds, ys)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            pbar.update(1)

    metrics["loss"] /= len(dataloader)
    metrics["accuracy"] /= len(dataloader)

    return metrics


def test_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> float:

    model.eval()
    with torch.inference_mode():
        metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
        }

        with tqdm(total=len(dataloader), leave=False, desc="Test") as pbar:
            for xs, ys in dataloader:
                xs, ys = xs.to(device), ys.to(device)

                with torch.autocast(device_type=device, enabled=use_amp, dtype=torch.float16):
                    preds = model(xs)
                    loss = loss_fn(preds, ys)

                metrics["loss"] += loss.item()
                metrics["accuracy"] += pixel_accuracy(preds, ys)

                pbar.update(1)

        metrics["loss"] /= len(dataloader)
        metrics["accuracy"] /= len(dataloader)

    return metrics


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

            train_metrics = train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )

            test_metrics = test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device,
                use_amp=scaler.is_enabled(),
            )

            if writer:
                # TODO: Add wandb writer
                writer.add_scalars(
                    main_tag="Loss",
                    tag_scalar_dict={"train": train_metrics["loss"], "test": test_metrics["loss"]},
                    global_step=epoch,
                )

                writer.add_scalars(
                    main_tag="Accuracy",
                    tag_scalar_dict={"train": train_metrics["accuracy"], "test": test_metrics["accuracy"]},
                    global_step=epoch,
                )

                writer.add_image(
                    tag="Prediction-Target",
                    img_tensor=get_image_for_tensorboard(model, x_vis, y_vis[0]),
                    global_step=epoch,
                )

            else:
                print(f"Epoch: {epoch}\t|\tTrain Loss: {train_metrics['loss']}\t|\tTest Loss: {test_metrics['loss']}")

            if model_saver is not None:
                model_saver(
                    current_loss=test_metrics["loss"],
                    epoch=epoch,
                    model=model,
                    optim=optimizer,
                    loss_fn=loss_fn,
                    scaler=scaler,
                )

            pbar.update(1)

    if writer:
        writer.close()


def pixel_accuracy(preds: torch.Tensor, ys: torch.Tensor) -> float:
    segm = torch.argmax(preds, dim=1)
    correct = torch.sum(segm == ys)

    return correct / torch.numel(ys)


def get_image_for_tensorboard(model, x, y):

    model.eval()
    with torch.inference_mode():
        preds = model(x)

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    pred = torch.argmax(preds[0, :, :, :], dim=0)
    axes[0].imshow(pred.cpu().numpy())
    axes[0].axis(False)

    axes[1].imshow(y.numpy())
    axes[1].axis(False)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    tensor = pil_to_tensor(Image.open(io.BytesIO(buf.getvalue())))

    plt.close()

    return tensor
