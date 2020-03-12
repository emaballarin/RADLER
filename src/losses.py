# ---------------------------------------------------------------------------- #
#                                                                              #
# RADLER ~ (adversarially) Robust Adversarial Distributional LEaRner           #
#                                                                              #
# |> Collection of NN losses, in PyTorch <|                                    #
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

import weights_util as wutil
import attacks_util as autil
import mnistnet_util as mutil
import architectures as myarchs


# ---------------------------------- #
# EMPIRICAL ACCURACY/ROBUSTNESS LOSS #
# ---------------------------------- #


def earloss(alpha, model, batchsize, th_device="cuda"):
    if (alpha < 0) or (alpha > 1):
        raise Exception('earloss: "alpha" parameter must be >= 0 and <= 1')
    return alpha * mutil.MNISTtest(model, batchsize, th_device) + (
        1.0 - alpha
    ) * autil.egrob_advatk_mnist(model, batchsize, th_device)


# -------------------------------------------------------- #
# MNISTNET EMPIRICAL ACCURACY/ROBUSTNESS LOSS FROM WEIGHTS #
# -------------------------------------------------------- #


def mnistnet_earloss_wfs(alpha, weights, batchsize, th_device):

    # Instantiate model
    fx_model = myarchs.SmallMNISTNet()

    # Load weights
    wutil.model_weightload(
        wutil.dictmodel_unflatten(weights, fx_model.state_dict()),
        fx_model,
        th_device,
        only_eval=True,
    )

    return earloss(alpha, fx_model, batchsize, th_device)
