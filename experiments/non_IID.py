import matplotlib.pyplot as plt
import torch

from src.data import (
    get_cifar10,
    create_dirichlet_clients
)
from src.model import CNN
from src.client import Client
from src.server import Server
from src.fedavg import train_fedavg
from src.utils import set_seed


# =========================================================
# Configuration
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 20
NUM_ROUNDS = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1
CLIENT_FRACTION = 1.0

ALPHA = 0.1
SEED = 42


print("Device:", DEVICE)
print("Alpha:", ALPHA)


# =========================================================
# Dataset
# =========================================================

train_dataset, _, _, test_loader = get_cifar10(
    batch_size=BATCH_SIZE
)


# =========================================================
# Non-IID partition
# =========================================================

client_datasets = create_dirichlet_clients(
    train_dataset,
    num_clients=NUM_CLIENTS,
    alpha=ALPHA,
    seed=SEED
)


# =========================================================
# Create clients
# =========================================================

clients = {}

for client_id, dataset in client_datasets.items():

    clients[client_id] = Client(
        client_id=client_id,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )


# =========================================================
# Global model
# =========================================================

set_seed(SEED)

global_model = CNN().to(DEVICE)


# =========================================================
# Server
# =========================================================

server = Server(
    model=global_model,
    clients=clients,
    client_fraction=CLIENT_FRACTION,
    seed=SEED
)


# =========================================================
# FedAvg
# =========================================================

history = train_fedavg(
    server=server,
    num_rounds=NUM_ROUNDS,
    test_loader=test_loader,
    device=DEVICE,
    epochs=LOCAL_EPOCHS,
    learning_rate=LEARNING_RATE
)


# =========================================================
# Final result
# =========================================================

print("\nFinal Result")

print(
    f"Non-IID α={ALPHA} | "
    f"Final Accuracy="
    f"{history['test_accuracy'][-1]:.4f}"
)


# =========================================================
# Accuracy
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["round"],
    history["test_accuracy"],
    label=f"α={ALPHA}"
)

plt.xlabel("Communication Round")
plt.ylabel("Test Accuracy")
plt.title("FedAvg under Non-IID Data")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/non_iid_alpha_01_accuracy.png"
)

plt.show()


# =========================================================
# Loss
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["round"],
    history["test_loss"],
    label=f"α={ALPHA}"
)

plt.xlabel("Communication Round")
plt.ylabel("Test Loss")
plt.title("FedAvg under Non-IID Data")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/non_iid_alpha_01_loss.png"
)

plt.show()