import torch

from src.data import get_cifar10, create_iid_clients
from src.model import CNN
from src.client import Client


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# Load dataset
train_dataset, _, _, _ = get_cifar10(
    batch_size=100
)


# Create clients
clients = create_iid_clients(
    train_dataset,
    num_clients=20,
    seed=42
)


# Create one client
client_dataset = clients[0]

client = Client(
    client_id=0,
    dataset=client_dataset,
    batch_size=100,
    device=device
)


# Create model
model = CNN()


# Train client
local_state, num_samples = client.train(
    model,
    epochs=1,
    learning_rate=0.01
)


print("Client ID:", client.client_id)
print("Number of samples:", num_samples)
print("Number of parameters:", len(local_state))