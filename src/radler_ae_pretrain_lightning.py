# ---------------------------------------------------------------------------- #
#                                                                              #
# RADLER ~ (adversarially) Robust Adversarial Distributional LEaRner           #
#                                                                              #
# |> Pretraining through AutoEncoding <|                                       #
#                                                                              #
# (C) 2019-* Emanuele Ballarin <emanuele@ballarin.cc>                          #
# (C) 2019-* AI-CPS@UniTS Laboratory (a.k.a. Bortolussi Group)                 #
#                                                                              #
# Distribution: MIT License                                                    #
# (Full text: https://github.com/emaballarin/RADLER/blob/master/LICENSE)       #
#                                                                              #
# Eventually-updated version: https://github.com/emaballarin/RADLER            #
#                                                                              #
# ---------------------------------------------------------------------------- #

# Adapted from A. Ansuini (cfr.: https://github.com/ansuini/MHPC_DL), S. Raschka
# (cfr.: https://github.com/rasbt/deeplearning-models/) and W. Falcon
# (cfr.: https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09)

# ------- #
# IMPORTS #
# ------- #

from __future__ import print_function

import os
import sys

import argparse

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader, random_split, TensorDataset

import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torchsummary import summary
from torchvision.utils import save_image

import pytorch_lightning as pl

# from src import radler_util as rutil
import radler_util as rutil


# ------- #
# DATASET #
# ------- #


# -------------------- #
# NETWORK ARCHITECTURE #
# -------------------- #

# Architecture: variation and deep-ification over S. Raschka's one
# (cfr.: https://github.com/rasbt/deeplearning-models)

# ENCODER (Scaffold):
class AE_Encoder(nn.Module):
    def __init__(self, data_size=20522, code_size=2):
        super(AE_Encoder, self).__init__()

        self.data_size = data_size
        self.code_size = code_size

        self.fc1 = nn.Linear(self.data_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, self.code_size)

    def forward(self, x):

        # Boilerplate
        # batch_size, channels, width, height = x.sizes()
        # x = x.view(batch_size, -1)

        # Layer 1
        x = self.fc1(x)
        x = F.leaky_relu(x)

        # Layer 2
        x = self.fc2(x)
        x = F.leaky_relu(x)

        # Layer 3
        x = self.fc3(x)
        x = F.leaky_relu(x)

        # Layer 4
        x = self.fc4(x)

        # Generated code
        code = F.tanh(x)  # We want our code to be "probably also negative"

        return code


# DECODER (Scaffold):
class AE_Decoder(nn.Module):
    def __init__(self, code_size=2, data_size=20522):
        super(AE_Decoder, self).__init__()

        self.code_size = code_size
        self.data_size = data_size

        self.fc1 = nn.Linear(self.code_size, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, self.data_size)

    def forward(self, x):

        # Boilerplate
        # batch_size, channels, width, height = x.sizes()
        # x = x.view(batch_size, -1)

        # Layer 1
        x = self.fc1(x)
        x = F.relu(x)

        # Layer 2
        x = self.fc2(x)
        x = F.relu(x)

        # Layer 3
        x = self.fc3(x)
        x = F.relu(x)

        # Layer 4
        x = self.fc4(x)

        # Generated output
        code = F.tanh(x)  # We want our data to be "probably also negative"

        return code


# AUTOENCODER (PyTorch Lightning module; just by putting E & D scaffolds together)
class Autoencoder(pl.LightningModule):
    def __init__(self, data_size=20522, code_size=2):
        super(Autoencoder, self).__init__()

        self.data_size = data_size
        self.code_size = code_size

        self.encoder = AE_Encoder(data_size, code_size)
        self.decoder = AE_Decoder(code_size, data_size)

    def forward(self, x):

        # Boilerplate
        # batch_size, channels, width, height = x.sizes()
        # x = x.view(batch_size, -1)

        # Encoding module
        x = self.encoder(x)

        # Decoding module
        x = self.decoder(x)

        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, code):
        return self.decoder(code)

    def MSE_loss(self, given_in, given_out):
        return (torch.nn.MSELoss(reduction="sum"))(given_in, given_out)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch, train_batch
        copied_input = self.forward(x)
        loss = self.MSE_loss(copied_input, y)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch, val_batch
        copied_input = self.forward(x)
        loss = self.MSE_loss(copied_input, y)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    # Stupid dataloader
    def stupid_dataloader(self):
        mydict = th.load("mnist_cnn_small.pt")
        mytensor = rutil.dictmodel_flatten(mydict, th_device="cuda")
        # return ((mytensor.clone().detach()).repeat(10000, 1).t()).t()
        return mytensor.clone().detach().repeat(1000, 1, 1)

    def train_dataloader(self):
        my_dataset = data.TensorDataset(self.stupid_dataloader())
        my_train = DataLoader(my_dataset, batch_size=32)
        return my_train

    def val_dataloader(self):
        my_dataset = data.TensorDataset(self.stupid_dataloader())
        my_val = DataLoader(my_dataset, batch_size=32)
        return my_val

    def test_dataloader(self):
        my_dataset = data.TensorDataset(self.stupid_dataloader())
        my_test = DataLoader(my_dataset, batch_size=32)
        return my_test

    def configure_optimizers(self):
        # the lightningModule HAS the parameters
        # (remember that we had the __init__ and forward method but we're just not showing it here)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer


# train
model = Autoencoder()
trainer = pl.Trainer(max_epochs=3)

trainer.fit(model)
