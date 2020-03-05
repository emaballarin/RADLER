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

# Adapted from A. Ansuini (cfr.: https://github.com/ansuini/MHPC_DL)

# ------- #
# IMPORTS #
# ------- #

from __future__ import print_function

import os
import sys

import torch as th
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.utils import save_image
import poutyne
from poutyne.framework import Model
from src import radler_util as rutil


# -------------------- #
# NETWORK ARCHITECTURE #
# -------------------- #

# Architecture: variation and deep-ification over S. Raschka's one
# (cfr.: https://github.com/rasbt/deeplearning-models)

# ENCODER:
class AE_encoder(nn.Module):
    def __init__(self, data_size=20522, code_size=2):
        super(AE_encoder, self).__init__()
        self.data_size = data_size
        self.code_size = code_size

        self.fc1 = nn.Linear(self.data_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, self.code_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        code = F.tanh(x)  # We want our code to be "probably also negative"
        return code


# DECODER:
class AE_decoder(nn.Module):
    def __init__(self, code_size=2, data_size=20522):
        super(AE_decoder, self).__init__()
        self.code_size = code_size
        self.data_size = data_size

        self.fc1 = nn.Linear(self.code_size, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, self.data_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        code = F.tanh(x)  # We want our data to be "probably also negative"
        return code


# AUTOENCODER (just by putting E & D together)
class autoencoder(nn.Module):
    def __init__(self, data_size=20522, code_size=2):
        super(autoencoder, self).__init__()
        self.code_size = code_size

        self.encoder = nn.Sequential(AE_encoder(data_size, code_size))
        self.decoder = nn.Sequential(AE_decoder(code_size, data_size))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, code):
        return self.decoder(code)


# -------------- #
# TRAINING SETUP #
# -------------- #

network = autoencoder()
optimizer = torch.optim.Adam(network.parameters(), lr=0.005, weight_decay=1e-5)
criterion = nn.MSELoss()


# ------------ #
# DATA LOADING #
# ------------ #
only_data_tensor = th.tensor(
    [
        rutil.dictmodel_flatten(th.load("mnist_cnn_small.pt")),
        rutil.dictmodel_flatten(th.load("mnist_cnn_small.pt")),
    ]
)

train_dataset = TensorDataset(only_data_tensor, only_data_tensor)
train_generator = DataLoader(train_dataset, batch_size=1)

valid_dataset = TensorDataset(only_data_tensor, only_data_tensor)
valid_generator = DataLoader(valid_dataset, batch_size=1)


# -------- #
# TRAINING #
# -------- #
model = Model(network, "sgd", "cross_entropy", batch_metrics=["accuracy"])
model.fit(
    train_generator, valid_generator, epochs=1, batch_size=1,
)
