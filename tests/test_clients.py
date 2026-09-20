from src.data import create_iid_clients, get_cifar10  

train_dataset, test_dataset, _, _ = get_cifar10(
    batch_size=100
)

clients = create_iid_clients(
    train_dataset,
    num_clients=20,
    seed=42
)


print("Number of clients:", len(clients))

for client_id, client_dataset in clients.items():

    print(
        f"Client {client_id}: "
        f"{len(client_dataset)} samples"
    )