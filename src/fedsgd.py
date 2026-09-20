import torch


def aggregate_gradients(
    global_model,
    local_gradients,
    client_sizes
):
    """
    Aggregate client gradients using weighted averaging.

    Args:
        global_model: Global model.
        local_gradients: List of client gradient dictionaries.
        client_sizes: Number of samples for each client.

    Returns:
        Aggregated gradients.
    """

    total_samples = sum(client_sizes)

    aggregated_gradients = {}

    for name, parameter in global_model.named_parameters():

        aggregated_gradients[name] = torch.zeros_like(
            parameter
        )

    for gradients, client_size in zip(
        local_gradients,
        client_sizes
    ):

        weight = client_size / total_samples

        for name in aggregated_gradients:

            aggregated_gradients[name] += (
                weight * gradients[name]
            )

    return aggregated_gradients


def apply_gradient_update(
    model,
    gradients,
    learning_rate
):
    """
    Apply one global gradient descent update.

    w_{t+1} = w_t - eta * g_t
    """

    with torch.no_grad():

        for name, parameter in model.named_parameters():

            parameter -= learning_rate * gradients[name]


def train_fedsgd(
    server,
    num_rounds,
    test_loader,
    device,
    learning_rate=0.01
):
    """
    Train the global model using FedSGD.

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

        local_gradients = []
        client_sizes = []

        # -------------------------
        # Compute client gradients
        # -------------------------

        for client_id in selected_ids:

            gradients, num_samples = server.clients[
                client_id
            ].compute_gradients(
                server.global_model
            )

            local_gradients.append(gradients)
            client_sizes.append(num_samples)

        # -------------------------
        # Aggregate gradients
        # -------------------------

        global_gradients = aggregate_gradients(
            server.global_model,
            local_gradients,
            client_sizes
        )

        # -------------------------
        # Global update
        # -------------------------

        apply_gradient_update(
            server.global_model,
            global_gradients,
            learning_rate
        )

        # -------------------------
        # Evaluation
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