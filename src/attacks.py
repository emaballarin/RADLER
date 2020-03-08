# ---------------------------------------------------------------------------- #
#                                                                              #
# RADLER ~ (adversarially) Robust Adversarial Distributional LEaRner           #
#                                                                              #
# |> Adversarially-attacking MNISTNet <|                                       #
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

# Adapted from G.W. Ding, Royal Bank of Canada (Borealis AI) et al.:
# https://github.com/BorealisAI/advertorch/blob/master/advertorch_examples/tutorial_attack_defense_bpda_mnist.ipynb

# ------- #
# IMPORTS #
# ------- #


import matplotlib.pyplot as plt

import os
import argparse

import torch
import torch as th

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow

# ATTACKS:
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import CarliniWagnerL2Attack

if __name__ == "__main__":
    import architectures as myarchs
    import weights_util as wutil
else:
    from src import architectures as myarchs
    from src import weights_util as wutil


# -------------------------- #
# PYTORCH DEVICE BOILERPLATE #
# -------------------------- #

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# ---------- #
# MODEL LOAD #
# ---------- #

loaddict = th.load("mnist_cnn_small.pt")
# loaddict = th.load("mnist_cnn.pt")

model = myarchs.SmallMNISTNet()
# model = myarchs.MNISTNet()
wutil.model_weightload(loaddict, model, th_device="cuda")  # Already in eval mode

# --------- #
# DATA LOAD #
# --------- #

batch_size = 250
loader = get_mnist_test_loader(batch_size=batch_size)
for cln_data, true_label in loader:
    break
cln_data, true_label = cln_data.to(device), true_label.to(device)

# -------------------- #
# INSTANTIATE ATTACKER #
# -------------------- #

adversary = LinfPGDAttack(
    model.forward,
    loss_fn=None,
    eps=0.175,
    nb_iter=160,
    eps_iter=0.00225,
    rand_init=True,
    clip_min=0.0,
    clip_max=1.0,
    targeted=False,
)

# adversary = CarliniWagnerL2Attack(
#     model.forward,
#     10,
#     confidence=0.1,
#     targeted=False,
#     learning_rate=0.01,
#     binary_search_steps=9,
#     max_iterations=10000,
#     abort_early=True,
#     initial_const=0.001,
#     clip_min=0.0,
#     clip_max=1.0,
#     loss_fn=None,
# )

# -------------------- #
# ATTACK! (UNTARGETED) #
# -------------------- #
adv_untargeted = adversary.perturb(cln_data, true_label)
pred_untargeted_adv = predict_from_logits(model(adv_untargeted))


# ------------------ #
# ATTACK! (TARGETED) #
# ------------------ #
# target = torch.ones_like(true_label) * 1  # Here "3" is the target!
# adversary.targeted = True
# adv_targeted = adversary.perturb(cln_data, target)
# pred_targeted_adv = predict_from_logits(model(adv_targeted))

# ------------ #
# SHOW ATTACKS #
# ------------ #

pred_cln = predict_from_logits(model(cln_data))

if batch_size <= 5:
    plt.figure(figsize=(10, 8))
    for ii in range(batch_size):
        plt.subplot(3, batch_size, ii + 1)
        _imshow(cln_data[ii])
        plt.title("clean \n pred: {}".format(pred_cln[ii]))
        plt.subplot(3, batch_size, ii + 1 + batch_size)
        _imshow(adv_untargeted[ii])
        plt.title("untargeted \n adv \n pred: {}".format(pred_untargeted_adv[ii]))

    plt.tight_layout()
    plt.show()

# --------- #
# FUNCTIONS #
# --------- #


def egrob(unpert_pred: th.Tensor, pert_pred: th.Tensor):
    return (
        th.sum((unpert_pred == pert_pred).clone().detach().double()).double()
        / th.tensor(pred_cln.size()).double()
    ).double()


print(egrob(pred_cln, pred_untargeted_adv))
