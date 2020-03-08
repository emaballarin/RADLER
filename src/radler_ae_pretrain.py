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

import torch
import torch as th
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import pytorch_lightning as pl

if __name__ == "__main__":
    import architectures as myarchs
    import weights_util as wutil
else:
    from src import architectures as myarchs
    from src import weights_util as wutil


# -------------------- #
# NETWORK ARCHITECTURE #
# -------------------- #

# Building blocks implemented in another module


# ------------------- #
# DATASET BOILERPLATE #
# ------------------- #

# A fake dataset composed all of the same in-out pair, which is the only example (to be autoencoded)
class MyStupidDataset(Dataset):
    def __init__(self):
        # Such example:
        self.my_single_example = (
            wutil.dictmodel_flatten(th.load("mnist_cnn_small.pt"), th_device="cuda")
            .clone()
            .detach()
        )

    def __getitem__(self, index):
        return self.my_single_example, self.my_single_example

    def __len__(self):
        return 1000


# AUTOENCODER (PyTorch Lightning module; just by putting E & D scaffolds together)
class Autoencoder(pl.LightningModule):
    def __init__(self, data_size=20522, code_size=2):
        super(Autoencoder, self).__init__()

        self.data_size = data_size
        self.code_size = code_size

        self.encoder = myarchs.AE_Encoder(data_size, code_size)
        self.decoder = myarchs.AE_Decoder(code_size, data_size)

    def forward(self, x):

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
        x, y = train_batch
        copied_input = self.forward(x)
        loss = self.MSE_loss(copied_input, y)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        copied_input = self.forward(x)
        loss = self.MSE_loss(copied_input, y)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def train_dataloader(self):
        my_dataset = MyStupidDataset()
        my_train = DataLoader(my_dataset, shuffle=False, batch_size=64)
        return my_train

    def val_dataloader(self):
        my_dataset = MyStupidDataset()
        my_val = DataLoader(my_dataset, shuffle=False, batch_size=64)
        return my_val

    def test_dataloader(self):
        my_dataset = MyStupidDataset()
        my_test = DataLoader(my_dataset, shuffle=False, batch_size=64)
        return my_test

    def configure_optimizers(self):
        # The lightningModule HAS the parameters
        # (remember that we had the __init__ and forward method but we're just not showing it here)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer


# train
model = Autoencoder()
trainer = pl.Trainer(
    # We are forcing pure overfitting here; still
    # we don't want too much of it.
    max_epochs=20,  # Autostopped, eventually
    gpus=1,
)

trainer.fit(model)

# For some curious reason (a.k.a. purposeful overfitting), the decoding part is
# almost centered at [0.0, 0.0] to produce the best replication of the in-out


# Save model (decoding part)
th.save(model.decoder.state_dict(), "bottleneck.pt")
