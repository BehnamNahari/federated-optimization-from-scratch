import copy


def aggregate_models(
    global_model,
    local_states,
    client_sizes
):
    """
    Perform weighted FedAvg aggregation.

    w_{t+1} = sum_k (n_k / n) * w_{t+1}^k
    """

    total_samples = sum(client_sizes)

    global_state = copy.deepcopy(
        local_states[0]
    )

    for key in global_state:

        global_state[key] = (
            global_state[key] * (
                client_sizes[0] / total_samples
            )
        )

    for local_state, client_size in zip(
        local_states[1:],
        client_sizes[1:]
    ):

        weight = client_size / total_samples

        for key in global_state:

            global_state[key] += (
                weight * local_state[key]
            )

    global_model.load_state_dict(
        global_state
    )



def train_fedavg(
    server,
    num_rounds,
    test_loader,
    device,
    epochs=1,
    learning_rate=0.01
):
    """
    Train the global model using FedAvg.

    Args:
        server: Server instance.
        num_rounds: Number of communication rounds.
        test_loader: Global test dataloader.
        device: Training device.
        epochs: Number of local epochs (E).
        learning_rate: Local SGD learning rate.

    Returns:
        history: Dictionary containing round-wise metrics.
    """

    from src.evaluate import evaluate

    history = {
        "round": [],
        "test_loss": [],
        "test_accuracy": []
    }

    for round_number in range(1, num_rounds + 1):

        # -------------------------
        # Select clients
        # -------------------------

        selected_ids = server.select_clients()

        local_states = []
        client_sizes = []

        # -------------------------
        # Local training
        # -------------------------

        for client_id in selected_ids:

            local_state, num_samples = server.clients[
                client_id
            ].train(
                server.global_model,
                epochs=epochs,
                learning_rate=learning_rate
            )

            local_states.append(local_state)
            client_sizes.append(num_samples)

        # -------------------------
        # FedAvg aggregation
        # -------------------------

        aggregate_models(
            server.global_model,
            local_states,
            client_sizes
        )

        # -------------------------
        # Global evaluation
        # -------------------------

        test_loss, test_accuracy = evaluate(
            server.global_model,
            test_loader,
            device
        )

        history["round"].append(round_number)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        print(
            f"Round {round_number}/{num_rounds} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.4f}"
        )

    return history