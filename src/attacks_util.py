# ---------------------------------------------------------------------------- #
#                                                                              #
# RADLER ~ (adversarially) Robust Adversarial Distributional LEaRner           #
#                                                                              #
# |> Utilities for dversarially-attacking MNISTNet <|                          #
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

import torch
import torch as th

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader

# ATTACK(S):
from advertorch.attacks import LinfPGDAttack

import architectures as myarchs
import weights_util as wutil

# ------------------- #
# UTILITY FUNCTION(S) #
# ------------------- #


def egrob_advatk_mnist(model, batch_size, th_device="cuda"):

    # Enforce correct device usage
    if (th_device != "cpu") and (th_device != "cuda"):
        raise Exception(
            'advatk_model: th_device specified is neither "cpu" nor "cuda".'
        )

    if th_device == "cuda" and not th.cuda.is_available():
        raise Exception(
            'advatk_model: th_device specified as "cuda", but no CUDA device available.'
        )

    # DATA LOADER:
    # batch_size = 250
    loader = get_mnist_test_loader(batch_size=250, shuffle=True)
    # loader = given_loader
    # print(type(loader))

    for cln_data, true_label in loader:
        break
    cln_data, true_label = cln_data.to(th_device), true_label.to(th_device)

    # ATTACKER INSTANTIATION:
    adversary = LinfPGDAttack(  # Change the attack name/params if needed!
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

    # PREDICT:
    pred_cln = predict_from_logits(model(cln_data))
    adv_untargeted = adversary.perturb(cln_data, true_label)
    pred_untargeted_adv = predict_from_logits(model(adv_untargeted))

    # COMPARE
    def egacc(unpert_pred: th.Tensor, true_labels: th.Tensor):
        return (
            th.sum((unpert_pred == true_labels).clone().detach().double()).double()
            / th.tensor(pred_cln.size()).double()
        ).double()

    def egrob(pert_pred: th.Tensor, true_labels: th.Tensor):
        return (
            th.sum((pert_pred == true_labels).clone().detach().double()).double()
            / th.tensor(pred_cln.size()).double()
        ).double()

    # RETURN
    return egacc(pred_cln, true_label), egrob(pred_untargeted_adv, true_label)
