from src.model import CNN


# =========================================================
# Configuration
# =========================================================

NUM_CLIENTS = 20
NUM_ROUNDS = 50

CLIENT_FRACTIONS = [
    0.10,
    0.25,
    0.50,
    1.00
]

BYTES_PER_PARAMETER = 4  # float32


# =========================================================
# Model size
# =========================================================

model = CNN()

num_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

model_size_bytes = (
    num_parameters
    * BYTES_PER_PARAMETER
)


# =========================================================
# Print model information
# =========================================================

print("Number of parameters:", num_parameters)

print(
    f"Model size: "
    f"{model_size_bytes / 1_000_000:.2f} MB"
)


# =========================================================
# Communication cost
# =========================================================

print("\nCommunication Cost")

for client_fraction in CLIENT_FRACTIONS:

    num_selected = max(
        1,
        int(client_fraction * NUM_CLIENTS)
    )

    download_per_round = (
        num_selected
        * model_size_bytes
    )

    upload_per_round = (
        num_selected
        * model_size_bytes
    )

    total_per_round = (
        download_per_round
        + upload_per_round
    )

    total_communication = (
        total_per_round
        * NUM_ROUNDS
    )

    print(
        f"\nC={client_fraction:.2f}"
    )

    print(
        f"Selected clients/round: "
        f"{num_selected}"
    )

    print(
        f"Download/round: "
        f"{download_per_round / 1_000_000:.2f} MB"
    )

    print(
        f"Upload/round: "
        f"{upload_per_round / 1_000_000:.2f} MB"
    )

    print(
        f"Total/round: "
        f"{total_per_round / 1_000_000:.2f} MB"
    )

    print(
        f"Total for {NUM_ROUNDS} rounds: "
        f"{total_communication / 1_000_000_000:.2f} GB"
    )