# ---------------------------------------------------------------------------- #
#                                                                              #
# RADLER ~ (adversarially) Robust Adversarial Distributional LEaRner           #
#                                                                              #
# |> Collection of NN architecture scaffolds, in PyTorch <|                    #
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


# ------- #
# IMPORTS #
# ------- #

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ------------- #
# MNIST CNN+FCN #
# ------------- #

# Chollet-Chintala-Ansuini architecture, modelled after:
# - F. Chollet et al. - Keras examples: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# - S. Chintala et al. - PyTorch examples: https://github.com/pytorch/examples/blob/master/mnist/
# - A. Ansuini et al. - Code for NeurIPS paper "Intrinsic Dimension of Data Representations in Deep Neural Networks":
#   https://github.com/ansuini/IntrinsicDimDeep/blob/master/mnist_archs.py


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.dropout1(x)
        x = th.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# ------------------- #
# SMALL MNIST CNN+FCN #
# ------------------- #

# Chollet-Chintala-Ansuini architecture, modelled after:
# - F. Chollet et al. - Keras examples: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# - S. Chintala et al. - PyTorch examples: https://github.com/pytorch/examples/blob/master/mnist/
# - A. Ansuini et al. - Code for NeurIPS paper "Intrinsic Dimension of Data Representations in Deep Neural Networks":
#   https://github.com/ansuini/IntrinsicDimDeep/blob/master/mnist_archs.py
#
# and shrunk in size with the aim of containing accuracy loss (avg. acc. loss 0.06)


class SmallMNISTNet(nn.Module):
    def __init__(self):
        super(SmallMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 1)
        self.conv2 = nn.Conv2d(8, 16, 5, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.dropout1(x)
        x = th.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# -------------------- #
# DEEP FCN AUTOENCODER #
# -------------------- #

# Multi-layer fully connected autoencoder, inspired by:
# - S. Raschka - Deep Learning Models Repository:
#   https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-basic.ipynb
# - A. Ansuini - Deep Learning lectures for the MHPC @ SISSA, Trieste:
#   https://github.com/ansuini/MHPC_DL/blob/master/day3/solutions/simple_autoencoder.ipynb

# ENCODER part:
class AE_Encoder(nn.Module):
    def __init__(self, data_size=20522, code_size=2, normalize=False):
        super(AE_Encoder, self).__init__()

        self.data_size = data_size
        self.code_size = code_size
        self.normalize = normalize

        self.fc1 = nn.Linear(self.data_size, 512)
        self.fc2 = nn.Linear(512, 128)
        if normalize:
            self.bn2 = nn.BatchNorm1d(128, 0.8)
        self.fc3 = nn.Linear(128, 32)
        if normalize:
            self.bn3 = nn.BatchNorm1d(32, 0.8)
        self.fc4 = nn.Linear(32, self.code_size)

    def forward(self, x):

        # Layer 1
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.07)

        # Layer 2
        x = self.fc2(x)
        if self.normalize:
            x = self.bn2(x)
        x = F.leaky_relu(x, 0.07)

        # Layer 3
        x = self.fc3(x)
        if self.normalize:
            x = self.bn3(x)
        x = F.leaky_relu(x, 0.07)

        # Layer 4
        x = self.fc4(x)

        # Generated code
        code = th.tanh(x)  # We want our code to be "probably also negative"

        return code


# DECODER part:
class AE_Decoder(nn.Module):
    def __init__(self, code_size=2, data_size=20522, normalize=False):
        super(AE_Decoder, self).__init__()

        self.code_size = code_size
        self.data_size = data_size
        self.normalize = normalize

        self.fc1 = nn.Linear(self.code_size, 32)
        self.fc2 = nn.Linear(32, 128)
        if normalize:
            self.bn2 = nn.BatchNorm1d(128, 0.8)
        self.fc3 = nn.Linear(128, 512)
        if normalize:
            self.bn3 = nn.BatchNorm1d(512, 0.8)
        self.fc4 = nn.Linear(512, self.data_size)

    def forward(self, x):

        # Layer 1
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.07)

        # Layer 2
        x = self.fc2(x)
        if self.normalize:
            x = self.bn2(x)
        x = F.leaky_relu(x, 0.07)

        # Layer 3
        x = self.fc3(x)
        if self.normalize:
            x = self.bn3(x)
        x = F.leaky_relu(x, 0.07)

        # Layer 4
        x = self.fc4(x)

        # Generated output
        data_out = th.tanh(x)  # We want our data to be "probably also negative"

        return data_out


# -------------------------- #
# LOSS-FUNCTION APPROXIMATOR #
# -------------------------- #

# A simple multi-layer fully connected "bottleneck", with one optional BatchNorm
# step


class LF_Approx(nn.Module):
    def __init__(self, data_size=20522, normalize=False):
        super(LF_Approx, self).__init__()

        self.data_size = data_size
        self.normalize = normalize

        self.fc1 = nn.Linear(self.data_size, 512)
        self.fc2 = nn.Linear(512, 32)
        if normalize:
            self.bn2 = nn.BatchNorm1d(512, 0.8)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):

        # Layer 1
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.02)

        # Layer 2
        x = self.fc2(x)
        if self.normalize:
            x = self.bn2(x)
        x = F.leaky_relu(x, 0.02)

        # Layer 3
        x = self.fc3(x)

        # Generated code
        estimate = 10 * th.sigmoid(
            x
        )  # We want our estimate to be positive, between 0 and 1

        return estimate
