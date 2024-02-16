from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

__all__ = [
    "get_dataloader",
]

def get_dataloader(batch_size: int, train: bool):
    data = datasets.FashionMNIST(
        root="data",
        train=train,
        download=True,
        transform=ToTensor(),
    )

    return DataLoader(data, batch_size=batch_size)
