# Experimental Results

This document records the principal results generated during the project.

## FedSGD vs. FedAvg

| Method | Final Test Accuracy |
|---|---:|
| FedSGD | 12.21% |
| FedAvg | 43.18% |

## Client Fraction

| C | Final Accuracy | Communication for 50 rounds |
|---:|---:|---:|
| 0.10 | 42.34% | 0.44 GB |
| 0.25 | 43.43% | 1.09 GB |
| 0.50 | 42.97% | 2.18 GB |
| 1.00 | 43.18% | 4.36 GB |

## Local Epochs

| E | Final Accuracy |
|---:|---:|
| 1 | 43.0% |
| 2 | 52.0% |
| 5 | 62.9% |
| 10 | 67.0% |

## Local Batch Size

| B | Final Accuracy |
|---:|---:|
| 50 | 50% |
| 100 | 43% |
| 250 | 33% |
| 500 | 25% |

## IID vs. Non-IID

| Setting | Final Accuracy |
|---|---:|
| IID | 43.18% |
| Non-IID, alpha=0.1 | 40.35% |

## Communication Cost

Model size: approximately 2.18 MB (FP32 payload).

| C | Clients/Round | Communication / Round | Total for 50 Rounds |
|---:|---:|---:|---:|
| 0.10 | 2 | 8.72 MB | 0.44 GB |
| 0.25 | 5 | 21.80 MB | 1.09 GB |
| 0.50 | 10 | 43.61 MB | 2.18 GB |
| 1.00 | 20 | 87.22 MB | 4.36 GB |

## Interpretation

The most important observations are:

1. FedAvg outperformed FedSGD in the measured communication-round budget.
2. Lower client participation dramatically reduced model communication in the IID experiment without materially changing final accuracy.
3. Increasing local epochs accelerated round-based convergence, at the cost of substantially more client-side computation.
4. Smaller local batches yielded better accuracy under the fixed-round protocol, partly because they produced more optimizer updates per local epoch.
5. Strong non-IID partitioning reduced FedAvg performance relative to the IID baseline.

These observations are specific to the implemented model, optimizer settings, dataset partitioning, and experimental protocol. They should not be interpreted as universal rankings of hyperparameters.
