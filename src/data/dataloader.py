import torch
import torchvision


def get_dataloaders(batch_size: int, num_workers: int):
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.RandomHorizontalFlip(p=0.5), torchvision.transforms.ToTensor()]
    )
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.Kitti(
        root="data", train=True, transform=train_transform, target_transform=None, download=True
    )

    test_dataset = torchvision.datasets.Kitti(
        root="data", train=False, transform=test_transform, target_transform=None, download=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_dataloader, test_dataloader, train_dataset.classes
