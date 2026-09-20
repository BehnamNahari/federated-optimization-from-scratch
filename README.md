# Federated Optimization from Scratch: FedSGD and FedAvg

A from-scratch experimental study of **Federated Stochastic Gradient Descent (FedSGD)** and **Federated Averaging (FedAvg)** under IID and non-IID data.

The project is inspired by:

> McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017.

The implementation intentionally avoids federated-learning frameworks such as Flower or FedML. The goal is to understand and reproduce the **core optimization mechanics** of federated learning directly with PyTorch, then study how client participation, local computation, minibatch size, and data heterogeneity affect convergence and communication.

---

## Highlights

- FedSGD implemented from scratch
- FedAvg implemented from scratch
- Manual client/server simulation on a single machine
- IID and Dirichlet-based non-IID data partitioning
- Controlled experiments over:
  - client fraction `C`
  - local epochs `E`
  - local batch size `B`
  - non-IID heterogeneity
- Communication-cost estimation
- Accuracy/loss curves and experiment summaries
- Reproducible experiment configuration with fixed seeds

---

## Project Goal

This project is not designed to maximize CIFAR-10 classification accuracy.

The primary goal is to understand and experimentally study
federated optimization, with a focus on the communication-efficiency
trade-offs introduced by FedSGD and FedAvg.

The implementation is inspired by McMahan et al. (2017), but it is
not an exact numerical reproduction of the original experiments.

---

## Research Questions

This project investigates four main questions:

1. **FedSGD vs. FedAvg:** How does local model optimization affect convergence compared with server-side gradient aggregation?
2. **Client participation:** What happens when only a fraction of clients participates in each communication round?
3. **Local computation:** How does increasing the number of local epochs affect convergence?
4. **Data heterogeneity:** How does non-IID client data affect FedAvg?

A secondary question is how the local minibatch size influences optimization under a fixed number of communication rounds.

---

## Method

### Federated setting

The training dataset is partitioned across simulated clients. In each communication round:

1. The server selects a subset of clients.
2. The current global model is sent to the selected clients.
3. Each client performs local optimization.
4. The clients return either gradients or updated model parameters.
5. The server aggregates the client updates.

The test set remains centralized and is used only to evaluate the global model.

---

## FedSGD

For selected clients \(S_t\), each client computes the gradient of its local objective:

\[
g_{k,t} = \nabla F_k(w_t)
\]

The server performs a sample-weighted aggregation:

\[
g_t =
\sum_{k \in S_t}
\frac{n_k}{\sum_{j \in S_t} n_j}
g_{k,t}
\]

and updates the global model:

\[
w_{t+1} = w_t - \eta g_t
\]

In this implementation, clients do **not** perform optimizer updates for FedSGD. They compute gradients and return them to the server.

---

## FedAvg

Each selected client starts from the current global model and performs local SGD:

\[
w_t
\rightarrow
w_{t,1}^k
\rightarrow
\dots
\rightarrow
w_{t,E}^k
\]

The server then performs weighted model averaging:

\[
w_{t+1}
=
\sum_{k \in S_t}
\frac{n_k}{\sum_{j \in S_t} n_j}
w_{t,E}^k
\]

The key idea is to perform more computation locally in exchange for fewer communication rounds.

---

## Dataset and Model

### Dataset

- CIFAR-10
- 50,000 training samples
- 10,000 test samples
- 10 classes
- 20 simulated clients

### Model

A lightweight convolutional neural network:

```text
Conv2d(3 → 32)
ReLU
MaxPool

Conv2d(32 → 64)
ReLU
MaxPool

Flatten
Linear(4096 → 128)
ReLU
Linear(128 → 10)
```

The model contains approximately **545K trainable parameters**.

At FP32, the model payload is approximately **2.18 MB**.

---

## Project Structure

```text
fedavg-from-scratch/
│
├── README.md
├── requirements.txt
├── main.py
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── client.py
│   ├── server.py
│   ├── fedsgd.py
│   ├── fedavg.py
│   ├── utils.py
│   └── results.py
│
├── tests/
│   ├── __init__.py
│   ├── test_clients.py
│   ├── test_client_training.py
│   ├── test_client_gradients.py
│   ├── test_server.py
│   ├── test_fedsgd.py
│   ├── test_fedavg.py
│   └── test_fedsgd_vs_fedavg.py
│
├── experiments/
│   ├── client_fraction.py
│   ├── local_epochs.py
│   ├── local_batch_size.py
│   ├── non_iid.py
│   ├── communication_cost.py
│   └── summary.py
│
└── results/
    ├── data/
    ├── figures/
    └── summary.md
```

---

# Experiments

All experiments use CIFAR-10 with 20 simulated clients unless otherwise stated.

## 1. FedSGD vs. FedAvg

Controlled baseline:

- 20 clients
- IID partition
- `C = 1.0`
- 50 communication rounds
- learning rate = `0.01`
- FedAvg local epochs = `1`
- local batch size = `100`

Final test accuracy from the recorded run:

| Method | Final Accuracy |
|---|---:|
| FedSGD | 12.21% |
| FedAvg | 43.18% |

### Interpretation

Under the same number of communication rounds, FedAvg substantially outperformed FedSGD in this setup.

The primary reason is the amount of local optimization performed before each communication step. FedSGD performs one global gradient update per round, whereas FedAvg performs multiple local SGD updates on each client before aggregation.

This result should be interpreted as a communication-round comparison, not as equal-computation or equal-wall-clock benchmarking.

---

## 2. Client Fraction \(C\)

Configuration:

- IID data
- `E = 1`
- batch size = `100`
- learning rate = `0.01`
- 50 rounds

| Client Fraction \(C\) | Selected Clients / Round | Final Accuracy |
|---:|---:|---:|
| 0.10 | 2 | 42.34% |
| 0.25 | 5 | 43.43% |
| 0.50 | 10 | 42.97% |
| 1.00 | 20 | 43.18% |

The four accuracy curves were broadly similar.

### Interpretation

Under IID data and equal-sized clients, reducing the participating fraction had relatively little effect on final accuracy in this particular simulation. Because clients have similar data distributions, a small random sample of clients can still provide a reasonable approximation of the overall training distribution.

This does **not** imply that small client fractions are universally harmless. Under stronger non-IID conditions, fewer participating clients can produce more variable and biased round updates.

---

## 3. Local Epochs \(E\)

Configuration:

- IID data
- `C = 1.0`
- batch size = `100`
- learning rate = `0.01`
- 50 rounds

| Local Epochs \(E\) | Final Accuracy |
|---:|---:|
| 1 | 43.0% |
| 2 | 52.0% |
| 5 | 62.9% |
| 10 | 67.0% |

### Interpretation

Increasing local epochs substantially improved convergence when accuracy was measured against the number of communication rounds.

For a client with 2,500 samples and batch size 100:

\[
updates\ per\ epoch = 2500/100 = 25
\]

Therefore:

- `E = 1` → 25 local optimizer steps/client/round
- `E = 2` → 50
- `E = 5` → 125
- `E = 10` → 250

The experiment demonstrates the central FedAvg trade-off:

> **More local computation can reduce the amount of communication required to make progress.**

However, these settings do not have equal local computational cost. Therefore the experiment should not be interpreted as proof that larger `E` is always better.

---

## 4. Local Batch Size \(B\)

Configuration:

- IID data
- `C = 1.0`
- `E = 1`
- learning rate = `0.01`
- 50 rounds

| Local Batch Size \(B\) | Final Accuracy |
|---:|---:|
| 50 | 50% |
| 100 | 43% |
| 250 | 33% |
| 500 | 25% |

### Interpretation

Smaller local minibatches produced better results in this experiment.

With 2,500 samples per client and one local epoch:

- `B = 50` → 50 optimizer steps
- `B = 100` → 25 steps
- `B = 250` → 10 steps
- `B = 500` → 5 steps

Thus, under a fixed number of communication rounds, smaller batches resulted in substantially more local parameter updates.

The result reflects both:
1. more frequent local optimization steps, and
2. greater stochasticity in minibatch gradients.

Therefore, it should not be interpreted as evidence that smaller batches are universally superior.

---

## 5. IID vs. Non-IID

For the non-IID experiment, client label distributions were generated using a Dirichlet partition.

For Dirichlet parameter:

\[
\alpha = 0.1
\]

the resulting client distributions were strongly heterogeneous.

Baseline:

- IID
- `C = 1.0`
- `E = 1`
- `B = 100`
- 50 rounds

Comparison:

| Setting | Final Accuracy |
|---|---:|
| IID | 43.18% |
| Non-IID, \(\alpha=0.1\) | 40.35% |

### Interpretation

The strongly heterogeneous partition reduced final accuracy by approximately **2.83 percentage points** in this setup.

The effect was moderate because all clients participated in every round and only one local epoch was used. More aggressive local training or lower participation can amplify the effect of client heterogeneity.

---

# Communication Cost

The current CNN has:

\[
545,098
\]

trainable parameters.

Assuming FP32:

\[
545,098 \times 4
\approx 2.18\text{ MB}
\]

The communication calculation counts the model payload sent in both directions:

```text
Server → Client
Client → Server
```

For 50 communication rounds:

| \(C\) | Clients/Round | Total Communication |
|---:|---:|---:|
| 0.10 | 2 | 0.44 GB |
| 0.25 | 5 | 1.09 GB |
| 0.50 | 10 | 2.18 GB |
| 1.00 | 20 | 4.36 GB |

This is a payload-level estimate and excludes transport/protocol overhead, compression, quantization, secure aggregation overhead, and other system-level costs.

### Key observation

The \(C=0.1\) setting used approximately one-tenth of the model communication of \(C=1.0\) over the same 50 rounds, while final accuracy in this IID experiment remained very similar.

This illustrates why client participation is a central systems-level trade-off in federated learning.

---

# Main Findings

### 1. FedAvg was substantially more effective than FedSGD per communication round

The recorded baseline reached:

```text
FedSGD  → 12.21%
FedAvg  → 43.18%
```

The main difference is the amount of local optimization performed between communication steps.

### 2. Client fraction had limited effect under IID data

Changing `C` from `1.0` to `0.1` produced similar final accuracy in this experiment, while dramatically reducing the amount of model communication.

### 3. More local epochs improved round-based convergence

Increasing `E` produced much faster progress per communication round in the IID setup.

The important caveat is that higher `E` also increases local computation substantially.

### 4. Smaller local batches performed better in this setup

Smaller `B` values yielded more optimizer steps per local epoch and produced better accuracy under the fixed-round experimental protocol.

### 5. Non-IID data degraded FedAvg

The \(\alpha=0.1\) Dirichlet partition reduced final accuracy relative to the IID baseline, demonstrating the sensitivity of federated optimization to client data heterogeneity.

---

# Limitations

This project is a **from-scratch experimental study inspired by the FedAvg paper**, not an exact numerical replication of the original 2017 experiments.

Important differences include:

- The implemented CNN is a lightweight custom model rather than the exact architecture used in every experiment of the paper.
- The current CIFAR-10 preprocessing is simpler than the full preprocessing pipeline described in the original paper.
- Experiments were simulated on a single machine rather than on physically distributed devices.
- Communication cost counts model payload bytes only.
- Wall-clock communication latency and system-level networking overhead are not modeled.
- The reported experiments use a single seed unless otherwise stated.
- The results are not intended to represent statistically robust averages over multiple independent runs.

These limitations are intentional: the primary objective of the repository is **understanding the optimization mechanics and empirically studying their behavior**, rather than reproducing every numerical result of the original paper.

---

# Reproducibility

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Run the basic tests:

```bash
python -m tests.test_clients
python -m tests.test_client_training
python -m tests.test_client_gradients
python -m tests.test_server
```

Run FedSGD/FedAvg experiments:

```bash
python -m tests.test_fedsgd
python -m tests.test_fedavg
python -m tests.test_fedsgd_vs_fedavg
```

Run parameter studies:

```bash
python -m experiments.client_fraction
python -m experiments.local_epochs
python -m experiments.local_batch_size
python -m experiments.non_iid
python -m experiments.communication_cost
```

Generate the summary:

```bash
python -m experiments.summary
```

---

# Future Work

The repository is designed as a foundation for more advanced federated optimization experiments.

Planned extensions include:

- FedProx
- SCAFFOLD
- FedNova
- stronger non-IID benchmarks
- multiple random seeds and confidence intervals
- communication compression
- client dropout and partial participation
- realistic asynchronous/straggler simulation
- personalized federated learning
- edge/IoT-oriented federated learning experiments

The natural next research direction is to compare **FedAvg, FedProx, SCAFFOLD, and FedNova under increasingly heterogeneous client data**.

---

# Reference

McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).

**Communication-Efficient Learning of Deep Networks from Decentralized Data.**

AISTATS 2017.

- Paper: https://proceedings.mlr.press/v54/mcmahan17a.html
- arXiv: https://arxiv.org/abs/1602.05629

---

## License

Add your preferred open-source license before publishing, e.g. MIT.
