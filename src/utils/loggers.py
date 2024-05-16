import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


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
                "loss": loss_fn,
                "scaler": scaler.state_dict(),
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
                    "loss": loss_fn,
                    "scaler": scaler.state_dict(),
                },
                f=best_path,
            )

            for old_file in old_files:
                if "best" in old_files:
                    old_files.remove(old_file)
                    os.remove(path=self.path / old_file)

        for old_file in old_files:
            os.remove(path=self.path / old_file)
