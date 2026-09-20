import matplotlib.pyplot as plt
from src.utils import set_seed
from src.data import get_cifar10, create_iid_clients
from src.model import CNN
from src.client import Client
from src.server import Server
from src.fedsgd import train_fedsgd
from src.fedavg import train_fedavg

device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

print("Device:", device)

NUM_CLIENTS = 20
NUM_ROUNDS = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.01
EPOCHS = 1
SEED = 42


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


def create_clients():
    clients = {}

    for client_id, dataset in client_datasets.items():

        clients[client_id] = Client(
            client_id=client_id,
            dataset=dataset,
            batch_size=BATCH_SIZE,
            device=device
        )

    return clients


# =========================================================
# FedSGD
# =========================================================

print("\n================ FedSGD ================\n")

set_seed(SEED)

fedsgd_clients = create_clients()

fedsgd_model = CNN().to(device)

fedsgd_server = Server(
    model=fedsgd_model,
    clients=fedsgd_clients,
    client_fraction=1.0,
    seed=SEED
)

fedsgd_history = train_fedsgd(
    server=fedsgd_server,
    num_rounds=NUM_ROUNDS,
    test_loader=test_loader,
    device=device,
    learning_rate=LEARNING_RATE
)


# =========================================================
# FedAvg
# =========================================================

print("\n================ FedAvg ================\n")

set_seed(SEED)

fedavg_clients = create_clients()

fedavg_model = CNN().to(device)

fedavg_server = Server(
    model=fedavg_model,
    clients=fedavg_clients,
    client_fraction=1.0,
    seed=SEED
)

fedavg_history = train_fedavg(
    server=fedavg_server,
    num_rounds=NUM_ROUNDS,
    test_loader=test_loader,
    device=device,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE
)


# =========================================================
# Final comparison
# =========================================================

print("\n================ Results ================\n")

print(
    f"FedSGD Final Accuracy: "
    f"{fedsgd_history['test_accuracy'][-1]:.4f}"
)

print(
    f"FedAvg Final Accuracy: "
    f"{fedavg_history['test_accuracy'][-1]:.4f}"
)


# -------------------------
# Accuracy comparison
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    fedsgd_history["round"],
    fedsgd_history["test_accuracy"],
    label="FedSGD"
)

plt.plot(
    fedavg_history["round"],
    fedavg_history["test_accuracy"],
    label="FedAvg"
)

plt.xlabel("Communication Round")
plt.ylabel("Test Accuracy")
plt.title("FedSGD vs FedAvg")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/fedsgd_vs_fedavg_accuracy.png"
)

plt.show()


# -------------------------
# Loss comparison
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    fedsgd_history["round"],
    fedsgd_history["test_loss"],
    label="FedSGD"
)

plt.plot(
    fedavg_history["round"],
    fedavg_history["test_loss"],
    label="FedAvg"
)

plt.xlabel("Communication Round")
plt.ylabel("Test Loss")
plt.title("FedSGD vs FedAvg")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/fedsgd_vs_fedavg_loss.png"
)

plt.show()