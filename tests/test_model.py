import torch
from src.model import CNN

model = CNN()

x = torch.randn(4, 3, 32, 32)

output = model(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)