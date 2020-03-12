# ---------------------------------------------------------------------------- #
#                                                                              #
# RADLER ~ (adversarially) Robust Adversarial Distributional LEaRner           #
#                                                                              #
# |> Utilities for the batch-evaluation of NNs trained on MNIST <|             #
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

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader

# ATTACK(S):
from advertorch.attacks import LinfPGDAttack

import architectures as myarchs
import weights_util as wutil


# ------------------- #
# UTILITY FUNCTION(S) #
# ------------------- #


def MNISTtest(model, batchsize, th_device="cuda"):

    # Enforce correct device usage
    if (th_device != "cpu") and (th_device != "cuda"):
        raise Exception('MNISTtest: th_device specified is neither "cpu" nor "cuda".')

    if th_device == "cuda" and not th.cuda.is_available():
        raise Exception(
            'MNISTtest: th_device specified as "cuda", but no CUDA device available.'
        )

    # DATA LOADER:
    batch_size = batchsize
    loader = get_mnist_test_loader(batch_size=batch_size, shuffle=True)
    for cln_data, true_label in loader:
        break
    cln_data, true_label = cln_data.to(th_device), true_label.to(th_device)

    # PREDICT:
    pred_cln = predict_from_logits(model(cln_data))

    # COMPARE
    def egacc(pred: th.Tensor, true: th.Tensor):
        return (
            th.sum((pred == true).clone().detach().double()).double()
            / th.tensor(pred_cln.size()).double()
        ).double()

    # RETURN
    return egacc(pred_cln, true_label)
