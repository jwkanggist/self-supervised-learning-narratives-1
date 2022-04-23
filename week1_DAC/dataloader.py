import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST


def retrieve_mnist(batch_size: int):
    train_data = MNIST(root="./data", train=True, download=True)
    test_data = MNIST(root="./data", train=False, download=True)

    data = torch.cat([train_data.data.float() / 255, test_data.data.float() / 255], 0)
    labels = torch.cat([train_data.targets, test_data.targets], 0)

    dataset = TensorDataset(data, labels)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
