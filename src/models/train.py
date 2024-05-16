import torch
from torch import nn
from tqdm import tqdm

import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor

import io

# import matplotlib.pyplot as plt
from PIL import Image


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

        # with torch.cuda.amp.autocast():
        preds = model(xs)

        preds = preds.transpose(1, 2).transpose(2, 3).contiguous().view(-1, preds.shape[1])
        ys = ys.view(-1)

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
            preds = preds.transpose(1, 2).transpose(2, 3).contiguous().view(-1, preds.shape[1])
            ys = ys.view(-1)

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
    writer=None,
    model_saver=None,
) -> None:

    visualize_x, visualize_y = next(iter(test_dataloader))
    visualize_x = visualize_x.to(device)

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

            model.eval()
            with torch.inference_mode():
                visualize_pred = model(visualize_x)

            fig, axes = plt.subplots(1, 2)
            visualize_pred, _ = torch.max(visualize_pred[0], dim=0, keepdim=True)
            axes[0].imshow(visualize_pred.permute(1, 2, 0).cpu().numpy())
            axes[0].axis(False)

            axes[1].imshow(visualize_y[0].numpy())
            axes[1].axis(False)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)

            writer.add_image(
                tag="Prediction-Target",
                img_tensor=pil_to_tensor(Image.open(io.BytesIO(buf.getvalue()))),
                global_step=epoch,
            )

            plt.close()

            # writer.add_graph(model=model, input_to_model=torch.randn(1, 3, 512, 512).to(device))

        if model_saver is not None:
            model_saver(
                current_loss=test_loss,
                epoch=epoch,
                model=model,
                optim=optimizer,
                loss_fn=loss_fn,
            )

    if writer:
        writer.close()
