import torch

from src.model import CNN
from src.data import get_cifar10
from src.train import train_one_epoch
from src.evaluate import evaluate


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


train_dataset, test_dataset, train_loader, test_loader = get_cifar10(
    batch_size=100
)


model = CNN().to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


num_epochs = 5


for epoch in range(num_epochs):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        device
    )

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc:.4f}"
    )