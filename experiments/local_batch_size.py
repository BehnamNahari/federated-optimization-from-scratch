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
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1
CLIENT_FRACTION = 1.0
SEED = 42

BATCH_SIZES = [50, 100, 250, 500]

print("Device:", DEVICE)


# =========================================================
# Dataset
# =========================================================

train_dataset, _, _, test_loader = get_cifar10(
    batch_size=100
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

def create_clients(batch_size):

    clients = {}

    for client_id, dataset in client_datasets.items():

        clients[client_id] = Client(
            client_id=client_id,
            dataset=dataset,
            batch_size=batch_size,
            device=DEVICE
        )

    return clients


# =========================================================
# Experiment
# =========================================================

results = {}


for batch_size in BATCH_SIZES:

    print("\n" + "=" * 50)
    print(f"B = {batch_size}")
    print("=" * 50)

    set_seed(SEED)

    clients = create_clients(batch_size)

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
        epochs=LOCAL_EPOCHS,
        learning_rate=LEARNING_RATE
    )

    results[batch_size] = history


# =========================================================
# Accuracy plot
# =========================================================

plt.figure(figsize=(8, 5))

for batch_size, history in results.items():

    plt.plot(
        history["round"],
        history["test_accuracy"],
        label=f"B={batch_size}"
    )

plt.xlabel("Communication Round")
plt.ylabel("Test Accuracy")
plt.title("Effect of Local Batch Size on FedAvg")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/local_batch_size_accuracy.png"
)

plt.show()


# =========================================================
# Loss plot
# =========================================================

plt.figure(figsize=(8, 5))

for batch_size, history in results.items():

    plt.plot(
        history["round"],
        history["test_loss"],
        label=f"B={batch_size}"
    )

plt.xlabel("Communication Round")
plt.ylabel("Test Loss")
plt.title("Effect of Local Batch Size on FedAvg")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/local_batch_size_loss.png"
)

plt.show()


# =========================================================
# Final results
# =========================================================

print("\nFinal Results")

for batch_size, history in results.items():

    final_accuracy = history["test_accuracy"][-1]

    print(
        f"B={batch_size:3d} | "
        f"Final Accuracy={final_accuracy:.4f}"
    )