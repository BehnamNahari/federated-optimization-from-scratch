# Experimental Summary

## FedSGD vs FedAvg

| Method | Final Accuracy |
|---|---:|
| FedSGD | 0.1221 |
| FedAvg | 0.4318 |

## Client Fraction

| C | Accuracy | Communication (GB) |
|---:|---:|---:|
| 0.10 | 0.4234 | 0.44 |
| 0.25 | 0.4343 | 1.09 |
| 0.50 | 0.4297 | 2.18 |
| 1.00 | 0.4318 | 4.36 |

## Local Epochs

| E | Accuracy |
|---:|---:|
| 1 | 0.4300 |
| 2 | 0.5200 |
| 5 | 0.6290 |
| 10 | 0.6700 |

## Local Batch Size

| B | Accuracy |
|---:|---:|
| 50 | 0.5000 |
| 100 | 0.4300 |
| 250 | 0.3300 |
| 500 | 0.2500 |

## IID vs Non-IID

| Setting | Accuracy |
|---|---:|
| IID | 0.4318 |
| Non-IID (alpha=0.1) | 0.4035 |
