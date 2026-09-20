import copy

import torch
from torch.utils.data import DataLoader

from src.train import train_one_epoch


class Client:

    def __init__(
        self,
        client_id,
        dataset,
        batch_size=100,
        device="cpu"
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

    def train(
        self,
        model,
        epochs=1,
        learning_rate=0.01
    ):
        model = copy.deepcopy(model)
        model.to(self.device)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate
        )

        for _ in range(epochs):

            train_one_epoch(
                model,
                self.dataloader,
                optimizer,
                self.device
            )

        return model.state_dict(), len(self.dataset)

    def compute_gradients(
        self,
        model
    ):
        model = copy.deepcopy(model)
        model.to(self.device)

        model.train()

        criterion = torch.nn.CrossEntropyLoss(
            reduction="sum"
        )

        model.zero_grad()

        total_samples = 0

        for images, labels in self.dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            total_samples += images.size(0)

        gradients = {}

        for name, parameter in model.named_parameters():

            if parameter.grad is not None:

                gradients[name] = (
                    parameter.grad.detach().clone()
                    / total_samples
                )

        return gradients, total_samples