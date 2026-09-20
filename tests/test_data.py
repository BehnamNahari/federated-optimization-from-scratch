from src.data import get_cifar10

train_dataset, test_dataset, train_loader, test_loader = get_cifar10(
    batch_size=100
)

print("Train samples:", len(train_dataset))
print("Test samples :", len(test_dataset))

images, labels = next(iter(train_loader))

print("Batch images:", images.shape)
print("Batch labels:", labels.shape)