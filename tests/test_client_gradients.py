import torch

from src.data import get_cifar10, create_iid_clients
from src.model import CNN
from src.client import Client

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# -------------------------
# Dataset
# -------------------------

train_dataset, _, _, _ = get_cifar10(
    batch_size=100
)


# -------------------------
# Create clients
# -------------------------

client_datasets = create_iid_clients(
    train_dataset,
    num_clients=20,
    seed=42
)


# -------------------------
# Create one client
# -------------------------

client = Client(
    client_id=0,
    dataset=client_datasets[0],
    batch_size=100,
    device=device
)


# -------------------------
# Create model
# -------------------------

model = CNN()


# -------------------------
# Compute gradients
# -------------------------

gradients, num_samples = client.compute_gradients(
    model
)


# -------------------------
# Inspect results
# -------------------------

print("Client ID:", client.client_id)
print("Number of samples:", num_samples)
print("Number of gradients:", len(gradients))


for name, gradient in gradients.items():

    print(
        f"{name}: "
        f"shape={gradient.shape}, "
        f"mean={gradient.mean().item():.6f}, "
        f"std={gradient.std().item():.6f}"
    )