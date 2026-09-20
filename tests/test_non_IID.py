from collections import Counter

from src.data import (
    get_cifar10,
    create_dirichlet_clients
)


train_dataset, _, _, _ = get_cifar10(
    batch_size=100
)


clients = create_dirichlet_clients(
    train_dataset,
    num_clients=20,
    alpha=0.1,
    seed=42
)


print("Number of clients:", len(clients))


for client_id, client_dataset in clients.items():

    labels = [
        train_dataset.targets[index]
        for index in client_dataset.indices
    ]

    distribution = Counter(labels)

    print(f"\nClient {client_id}")
    print("Total samples:", len(client_dataset))

    for class_id in range(10):

        print(
            f"Class {class_id}: "
            f"{distribution[class_id]}"
        )