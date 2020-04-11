import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import pytorch_lightning as pl


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch

        print(x)
        print(y)

        y_hat = self(x)
        return {"loss": F.cross_entropy(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(
            MNIST(
                os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
            ),
            batch_size=1,
        )

    def get_mnist_train_loader(batch_size, shuffle=True):
        loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                DATA_PATH, train=True, download=True, transform=transforms.ToTensor()
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )
        loader.name = "mnist_train"
        return loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


trainer = pl.Trainer()
model = LitModel()

trainer.fit(model)
