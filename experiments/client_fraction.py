import torch
import matplotlib.pyplot as plt

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
LOCAL_EPOCHS = 1
SEED = 42

CLIENT_FRACTIONS = [0.10, 0.25, 0.50, 1.00]


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


for client_fraction in CLIENT_FRACTIONS:

    print("\n" + "=" * 50)
    print(f"C = {client_fraction}")
    print("=" * 50)

    set_seed(SEED)

    clients = create_clients()

    model = CNN().to(DEVICE)

    server = Server(
        model=model,
        clients=clients,
        client_fraction=client_fraction,
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

    results[client_fraction] = history


# =========================================================
# Accuracy plot
# =========================================================

plt.figure(figsize=(8, 5))

for client_fraction, history in results.items():

    plt.plot(
        history["round"],
        history["test_accuracy"],
        label=f"C={client_fraction}"
    )

plt.xlabel("Communication Round")
plt.ylabel("Test Accuracy")
plt.title("Effect of Client Fraction on FedAvg")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/client_fraction_accuracy.png"
)

plt.show()


# =========================================================
# Loss plot
# =========================================================

plt.figure(figsize=(8, 5))

for client_fraction, history in results.items():

    plt.plot(
        history["round"],
        history["test_loss"],
        label=f"C={client_fraction}"
    )

plt.xlabel("Communication Round")
plt.ylabel("Test Loss")
plt.title("Effect of Client Fraction on FedAvg")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/client_fraction_loss.png"
)

plt.show()


# =========================================================
# Final results
# =========================================================

print("\nFinal Results")

for client_fraction, history in results.items():

    final_accuracy = history["test_accuracy"][-1]

    print(
        f"C={client_fraction:.2f} | "
        f"Final Accuracy={final_accuracy:.4f}"
    )