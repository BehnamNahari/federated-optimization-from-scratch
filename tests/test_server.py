import torch

from src.data import get_cifar10, create_iid_clients
from src.model import CNN
from src.client import Client
from src.server import Server


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


# -------------------------
# Create server
# -------------------------

server = Server(
    model=global_model,
    clients=clients,
    client_fraction=1.0
)


# -------------------------
# Select clients
# -------------------------

selected_ids = server.select_clients()

print("Selected clients:", selected_ids)
print("Number selected:", len(selected_ids))


# -------------------------
# Local training
# -------------------------

local_states = []
client_sizes = []

for client_id in selected_ids:

    local_state, num_samples = clients[client_id].train(
        server.global_model,
        epochs=1,
        learning_rate=0.01
    )

    local_states.append(local_state)
    client_sizes.append(num_samples)


# -------------------------
# Aggregation
# -------------------------

server.aggregate(
    local_states,
    client_sizes
)


print("Aggregation completed.")