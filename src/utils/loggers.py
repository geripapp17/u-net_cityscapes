import os
import io
from pathlib import Path

import torch
from torch import nn
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
import matplotlib.pyplot as plt


def create_tensorboard_writer(
    path: Path,
    experiment_name: str,
    model_name: str,
    extra: str = None,
) -> None:

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d")

    log_dir = path / timestamp / experiment_name / model_name
    if extra is not None:
        log_dir = log_dir / extra

    return SummaryWriter(log_dir=log_dir)


class ModelSaver:
    def __init__(self, path: str, model_name: str) -> None:
        self.best_loss = float("inf")

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name

    def __call__(
        self,
        current_loss: float,
        epoch: int,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
        scaler: torch.cuda.amp.GradScaler,
    ) -> None:

        old_files = os.listdir(self.path)

        best_path = self.path / f"{self.model_name}_epoch-{epoch}_best.pth"
        latest_path = self.path / f"{self.model_name}_epoch-{epoch}_latest.pth"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_state_dict": loss_fn,
                "scaler_state_dict": scaler.state_dict(),
            },
            f=latest_path,
        )

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss_state_dict": loss_fn,
                    "scaler_state_dict": scaler.state_dict(),
                },
                f=best_path,
            )

            for old_file in [n for n in old_files if "best" in n]:
                os.remove(path=self.path / old_file)

        for old_file in [n for n in old_files if "latest" in n]:
            os.remove(path=self.path / old_file)


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
