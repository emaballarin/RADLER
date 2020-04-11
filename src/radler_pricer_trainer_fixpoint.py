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

# Adapted from W. Falcon:
# https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09

# ------- #
# IMPORTS #
# ------- #

from __future__ import print_function

import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

from advertorch_examples.utils import get_mnist_test_loader
from advertorch.utils import predict_from_logits


if __name__ == "__main__":
    import architectures as myarchs
    import weights_util as wutil
    import attacks_util as autil
else:
    from src import architectures as myarchs
    from src import weights_util as wutil
    from src import attacks_util as autil

# -------------------- #
# NETWORK ARCHITECTURE #
# -------------------- #

# Building blocks implemented in another module


# ------------------- #
# DATASET BOILERPLATE #
# ------------------- #


# -------------------- #
# 1-STEP INVISIBLEHAND #
# -------------------- #


class GAN(LightningModule):
    def __init__(self):
        super().__init__()

        # networks
        self.R_generator_arch = myarchs.AE_Decoder()
        self.R_loss_approx = myarchs.LF_Approx()
        self.R_target_arch = myarchs.SmallMNISTNet()

        # Model-weights dictionary prototypes
        self.mnistnet_prototype = th.load("mnist_cnn_small.pt")
        self.bottleneck_prototype = th.load("bottleneck.pt")

        # Load pretraining weights
        wutil.model_weightload(
            self.bottleneck_prototype, self.R_generator_arch, "cuda", only_eval=False
        )
        # hyperparameters for training
        # self.batch_size_gen = 25
        # self.batch_size_approx = 25

    def forward(self, code):
        return self.R_generator_arch(code)

    def loss_nn_approx(self, weights_vec):
        return self.R_loss_approx.forward(weights_vec)

    def MSE_loss(self, given_in, given_out):
        return torch.nn.MSELoss(reduction="sum")(given_in, given_out)

    def training_step(self, batch, batch_idx):

        _ = batch  # Useless but necessary ;)

        # COMMON

        # Sample noise
        # gen_in = (2.0 * th.rand(2) - 1.0).cuda()
        # gen_in = 0.0 * (2.0 * th.rand(2) - 1.0).cuda()
        gen_in = th.tensor([0.0, 0.0]).cuda()

        # Pass it through the generator
        gen_out = self.R_generator_arch.forward(gen_in)

        # Compute the loss
        gen_loss_approx = self.R_loss_approx.forward(gen_out)

        # ~~ WEIGHTS MANIPULATIONS ~~:

        # Convert generator output as weights for the target architecture
        weight_dict = wutil.dictmodel_unflatten(gen_out, self.mnistnet_prototype)

        # Load weights to the target architecture
        wutil.model_weightload(weight_dict, self.R_target_arch, "cuda", only_eval=True)

        # ~~ TARGET USAGE ~~:

        egacc, egrob = autil.egrob_advatk_mnist(self.R_target_arch, 10000, "cuda")

        targetloss = 10 * th.tensor([egacc, egrob]).cuda()

        approx_loss = self.MSE_loss(targetloss, gen_loss_approx)

        tqdm_dict = {"approx_loss": approx_loss}
        output = OrderedDict(
            {"loss": approx_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

    def configure_optimizers(self):
        opt_approx = torch.optim.Adam(
            self.parameters(), lr=3.5 * 1e-1, weight_decay=1.5 * 1e-2
        )
        return opt_approx

    def train_dataloader(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])]
        )
        dataset = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        loader = DataLoader(dataset, shuffle=False, batch_size=10000)
        loader.name = "mnist_test"
        return loader


def main():
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = GAN()

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer(gpus=1, max_epochs=400)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)

    th.save(model.R_loss_approx.state_dict(), "pricer_fixpoint.pt")


main()
