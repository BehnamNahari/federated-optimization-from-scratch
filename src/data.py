import torch
import numpy as np
from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_cifar10(batch_size=100):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataset, test_dataset, train_loader, test_loader

def create_iid_clients(dataset, num_clients=20, seed=42):

    generator = torch.Generator()
    generator.manual_seed(seed)

    num_samples = len(dataset)

    indices = torch.randperm(
        num_samples,
        generator=generator
    ).tolist()

    samples_per_client = num_samples // num_clients

    clients = {}

    for client_id in range(num_clients):

        start = client_id * samples_per_client
        end = start + samples_per_client

        client_indices = indices[start:end]

        clients[client_id] = Subset(
            dataset,
            client_indices
        )

    return clients

def create_dirichlet_clients(
    dataset,
    num_clients=20,
    alpha=0.5,
    seed=42
):
    """
    Split a labeled dataset among clients using
    a Dirichlet distribution.

    Smaller alpha -> stronger Non-IID heterogeneity.
    """

    rng = np.random.default_rng(seed)

    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    class_indices = {
        class_id: np.where(labels == class_id)[0].tolist()
        for class_id in range(num_classes)
    }

    client_indices = defaultdict(list)

    for class_id in range(num_classes):

        indices = class_indices[class_id]

        rng.shuffle(indices)

        proportions = rng.dirichlet(
            np.full(num_clients, alpha)
        )

        split_points = (
            np.cumsum(proportions)
            * len(indices)
        ).astype(int)

        split_points = np.insert(
            split_points,
            0,
            0
        )

        for client_id in range(num_clients):

            start = split_points[client_id]
            end = split_points[client_id + 1]

            client_indices[client_id].extend(
                indices[start:end]
            )

    clients = {}

    for client_id in range(num_clients):

        clients[client_id] = Subset(
            dataset,
            client_indices[client_id]
        )

    return clients