import torch

from src.data import (
    get_cifar10,
    create_iid_clients
)
from src.model import CNN
from src.client import Client
from src.server import Server
from src.fedavg import train_fedavg


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# -------------------------
# Dataset
# -------------------------

train_dataset, _, _, test_loader = get_cifar10(
    batch_size=100
)


# -------------------------
# Create client datasets
# -------------------------

client_datasets = create_iid_clients(
    train_dataset,
    num_clients=20,
    seed=42
)


# -------------------------
# Create clients
# -------------------------

clients = {}

for client_id, dataset in client_datasets.items():

    clients[client_id] = Client(
        client_id=client_id,
        dataset=dataset,
        batch_size=100,
        device=device
    )


# -------------------------
# Create global model
# -------------------------

global_model = CNN()
global_model.to(device)


# -------------------------
# Create server
# -------------------------

server = Server(
    model=global_model,
    clients=clients,
    client_fraction=1.0
)


# -------------------------
# Initial evaluation
# -------------------------

from src.evaluate import evaluate

initial_loss, initial_accuracy = evaluate(
    server.global_model,
    test_loader,
    device
)

print(
    f"Initial | "
    f"Test Loss: {initial_loss:.4f} | "
    f"Test Acc: {initial_accuracy:.4f}"
)


# -------------------------
# FedAvg training
# -------------------------

history = train_fedavg(
    server=server,
    num_rounds=50,
    test_loader=test_loader,
    device=device,
    epochs=1,
    learning_rate=0.01
)


print("\nTraining completed.")

print(
    f"Final Test Accuracy: "
    f"{history['test_accuracy'][-1]:.4f}"
)