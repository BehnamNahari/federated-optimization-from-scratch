import matplotlib.pyplot as plt


# =========================================================
# Results collected from experiments
# =========================================================

client_fraction_results = {
    0.10: 0.4234,
    0.25: 0.4343,
    0.50: 0.4297,
    1.00: 0.4318,
}

communication_results = {
    0.10: 0.44,
    0.25: 1.09,
    0.50: 2.18,
    1.00: 4.36,
}

local_epochs_results = {
    1: 0.430,
    2: 0.520,
    5: 0.629,
    10: 0.670,
}

batch_size_results = {
    50: 0.50,
    100: 0.43,
    250: 0.33,
    500: 0.25,
}

fedsgd_vs_fedavg = {
    "FedSGD": 0.1221,
    "FedAvg": 0.4318,
}

iid_vs_non_iid = {
    "IID": 0.4318,
    "Non-IID (alpha=0.1)": 0.4035,
}


# =========================================================
# Print summary tables
# =========================================================

print("\n" + "=" * 60)
print("FEDSGD vs FEDAVG")
print("=" * 60)

for method, accuracy in fedsgd_vs_fedavg.items():

    print(
        f"{method:<10} | "
        f"Accuracy = {accuracy:.4f}"
    )


print("\n" + "=" * 60)
print("CLIENT FRACTION")
print("=" * 60)

print(
    f"{'C':<10}"
    f"{'Accuracy':<15}"
    f"{'Communication (GB)':<20}"
)

for c in client_fraction_results:

    print(
        f"{c:<10.2f}"
        f"{client_fraction_results[c]:<15.4f}"
        f"{communication_results[c]:<20.2f}"
    )


print("\n" + "=" * 60)
print("LOCAL EPOCHS")
print("=" * 60)

print(
    f"{'E':<10}"
    f"{'Accuracy':<15}"
)

for epochs, accuracy in local_epochs_results.items():

    print(
        f"{epochs:<10}"
        f"{accuracy:<15.4f}"
    )


print("\n" + "=" * 60)
print("LOCAL BATCH SIZE")
print("=" * 60)

print(
    f"{'B':<10}"
    f"{'Accuracy':<15}"
)

for batch_size, accuracy in batch_size_results.items():

    print(
        f"{batch_size:<10}"
        f"{accuracy:<15.4f}"
    )


print("\n" + "=" * 60)
print("IID vs NON-IID")
print("=" * 60)

for setting, accuracy in iid_vs_non_iid.items():

    print(
        f"{setting:<25} | "
        f"Accuracy = {accuracy:.4f}"
    )


# =========================================================
# Plot 1 — Client Fraction vs Accuracy
# =========================================================

plt.figure(figsize=(8, 5))

c_values = list(client_fraction_results.keys())
accuracy_values = list(client_fraction_results.values())

plt.plot(
    c_values,
    accuracy_values,
    marker="o"
)

plt.xlabel("Client Fraction (C)")
plt.ylabel("Final Test Accuracy")
plt.title("Client Fraction vs Final Accuracy")
plt.xticks(c_values)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/summary_c_vs_accuracy.png",
    dpi=300
)

plt.show()


# =========================================================
# Plot 2 — Client Fraction vs Communication
# =========================================================

plt.figure(figsize=(8, 5))

communication_values = [
    communication_results[c]
    for c in c_values
]

plt.plot(
    c_values,
    communication_values,
    marker="o"
)

plt.xlabel("Client Fraction (C)")
plt.ylabel("Communication Cost (GB)")
plt.title("Client Fraction vs Communication Cost")
plt.xticks(c_values)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/summary_c_vs_communication.png",
    dpi=300
)

plt.show()


# =========================================================
# Plot 3 — Local Epochs vs Accuracy
# =========================================================

plt.figure(figsize=(8, 5))

e_values = list(local_epochs_results.keys())
e_accuracy = list(local_epochs_results.values())

plt.plot(
    e_values,
    e_accuracy,
    marker="o"
)

plt.xlabel("Local Epochs (E)")
plt.ylabel("Final Test Accuracy")
plt.title("Local Epochs vs Final Accuracy")
plt.xticks(e_values)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/summary_e_vs_accuracy.png",
    dpi=300
)

plt.show()


# =========================================================
# Plot 4 — Batch Size vs Accuracy
# =========================================================

plt.figure(figsize=(8, 5))

b_values = list(batch_size_results.keys())
b_accuracy = list(batch_size_results.values())

plt.plot(
    b_values,
    b_accuracy,
    marker="o"
)

plt.xlabel("Local Batch Size (B)")
plt.ylabel("Final Test Accuracy")
plt.title("Local Batch Size vs Final Accuracy")
plt.xticks(b_values)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/figures/summary_b_vs_accuracy.png",
    dpi=300
)

plt.show()


# =========================================================
# Plot 5 — IID vs Non-IID
# =========================================================

plt.figure(figsize=(8, 5))

labels = list(iid_vs_non_iid.keys())
values = list(iid_vs_non_iid.values())

plt.bar(
    labels,
    values
)

plt.ylabel("Final Test Accuracy")
plt.title("IID vs Non-IID FedAvg")
plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    "results/figures/summary_iid_vs_non_iid.png",
    dpi=300
)

plt.show()


# =========================================================
# Save Markdown summary
# =========================================================

with open(
    "results/summary.md",
    "w",
    encoding="utf-8"
) as file:

    file.write("# Experimental Summary\n\n")

    file.write(
        "## FedSGD vs FedAvg\n\n"
        "| Method | Final Accuracy |\n"
        "|---|---:|\n"
    )

    for method, accuracy in fedsgd_vs_fedavg.items():

        file.write(
            f"| {method} | {accuracy:.4f} |\n"
        )

    file.write("\n## Client Fraction\n\n")

    file.write(
        "| C | Accuracy | Communication (GB) |\n"
        "|---:|---:|---:|\n"
    )

    for c in c_values:

        file.write(
            f"| {c:.2f} | "
            f"{client_fraction_results[c]:.4f} | "
            f"{communication_results[c]:.2f} |\n"
        )

    file.write("\n## Local Epochs\n\n")

    file.write(
        "| E | Accuracy |\n"
        "|---:|---:|\n"
    )

    for epochs, accuracy in local_epochs_results.items():

        file.write(
            f"| {epochs} | {accuracy:.4f} |\n"
        )

    file.write("\n## Local Batch Size\n\n")

    file.write(
        "| B | Accuracy |\n"
        "|---:|---:|\n"
    )

    for batch_size, accuracy in batch_size_results.items():

        file.write(
            f"| {batch_size} | {accuracy:.4f} |\n"
        )

    file.write("\n## IID vs Non-IID\n\n")

    file.write(
        "| Setting | Accuracy |\n"
        "|---|---:|\n"
    )

    for setting, accuracy in iid_vs_non_iid.items():

        file.write(
            f"| {setting} | {accuracy:.4f} |\n"
        )


print("\nSummary generated successfully.")
print("Figures saved to: results/figures/")
print("Summary saved to: results/summary.md")