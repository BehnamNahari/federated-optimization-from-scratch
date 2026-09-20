import matplotlib.pyplot as plt
import torch

from src.data import get_cifar10, create_iid_clients
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
CLIENT_FRACTION = 1.0
SEED = 42

LOCAL_EPOCHS = [1, 2, 5, 10]

print("Device:", DEVICE)


# =========================================================
# Dataset
# =========================================================

train_dataset, _, _, test_loader = get_cifar10(
    batch_size=BATCH_SIZE
)


# =========================================================
# Fixed IID partition
# =========================================================

client_datasets = create_iid_clients(
    train_dataset,
    num_clients=NUM_CLIENTS,
    seed=SEED
)


# =========================================================
# Client creation
# =========================================================

def create_clients():

    clients = {}

    for client_id, dataset in client_datasets.items():

        clients[client_id] = Client(
            client_id=client_id,
            dataset=dataset,
            batch_size=BATCH_SIZE,
            device=DEVICE
        )

    return clients


# =========================================================
# Experiment
# =========================================================

results = {}


for epochs in LOCAL_EPOCHS:

    print("\n" + "=" * 50)
    print(f"E = {epochs}")
    print("=" * 50)

    set_seed(SEED)

    clients = create_clients()

    model = CNN().to(DEVICE)

    server = Server(
        model=model,
        clients=clients,
        client_fraction=CLIENT_FRACTION,
        seed=SEED
    )

    history = train_fedavg(
        server=server,
        num_rounds=NUM_ROUNDS,
        test_loader=test_loader,
        device=DEVICE,
        epochs=epochs,
        learning_rate=LEARNING_RATE
    )

    results[epochs] = history


# =========================================================
# Accuracy plot
# =========================================================

plt.figure(figsize=(8, 5))

for epochs, history in results.items():

    plt.plot(
        history["round"],
        history["test_accuracy"],
        label=f"E={epochs}"
    )

plt.xlabel("Communication Round")
plt.ylabel("Test Accuracy")
plt.title("Effect of Local Epochs on FedAvg")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/local_epochs_accuracy.png"
)

plt.show()


# =========================================================
# Loss plot
# =========================================================

plt.figure(figsize=(8, 5))

for epochs, history in results.items():

    plt.plot(
        history["round"],
        history["test_loss"],
        label=f"E={epochs}"
    )

plt.xlabel("Communication Round")
plt.ylabel("Test Loss")
plt.title("Effect of Local Epochs on FedAvg")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/local_epochs_loss.png"
)

plt.show()


# =========================================================
# Final results
# =========================================================

print("\nFinal Results")

for epochs, history in results.items():

    final_accuracy = history["test_accuracy"][-1]

    print(
        f"E={epochs:2d} | "
        f"Final Accuracy={final_accuracy:.4f}"
    )