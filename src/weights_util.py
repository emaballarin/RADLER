# ---------------------------------------------------------------------------- #
#                                                                              #
# RADLER ~ (adversarially) Robust Adversarial Distributional LEaRner           #
#                                                                              #
# |> Utility functions <|                                                      #
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

import torch as th
import collections


# ---------------- #
# FUNCTION-ALIASES #
# ---------------- #


def same(lhs, rhs):
    return lhs == rhs


# ------------------- #
# AUXILIARY FUNCTIONS #
# ------------------- #


def dictmodel_numel(dictmodel: collections.OrderedDict):
    elems: int = 0
    for key in dictmodel:
        elems += dictmodel.get(key).numel()
    return elems


def dictmodel_flatten(dictmodel: collections.OrderedDict, th_device="cuda"):

    # Enforce correct device usage
    if (th_device != "cpu") and (th_device != "cuda"):
        raise Exception(
            'model_flatten: th_device specified is neither "cpu" nor "cuda".'
        )

    if th_device == "cuda" and not th.cuda.is_available():
        raise Exception(
                'model_flatten: th_device specified as "cuda", but no CUDA device available.'
        )

    flatmodel: th.Tensor = th.tensor([]).to(th_device)

    for key in dictmodel:
        to_add: th.Tensor = dictmodel.get(key).flatten().to(th_device)
        flatmodel = th.cat((flatmodel, to_add), dim=0)

    flatmodel = flatmodel.to(th_device)

    # Cautionary consistency check
    if flatmodel.size() != th.Size([dictmodel_numel(dictmodel)]):
        print(
            "model_flatten: Number-of-Elements mismatch between FLAT and DICT models!"
        )
        raise Exception(
            "THIS EXCEPTION IS A DIAGNOSTIC FEATURE. SHOULD NEVER HAPPEN! See printed message for errors."
        )

    return flatmodel


def dictmodel_unflatten(flatmodel: th.Tensor, as_dictmodel: collections.OrderedDict):
    # Consistency check
    if flatmodel.size() != th.Size([dictmodel_numel(as_dictmodel)]):
        raise Exception(
            "model_unflatten: Number-of-Elements mismatch between FLAT and AS_DICT models!"
        )

    dictmodel: dict = {}
    start: int = 0
    stop: int = 0

    for key in as_dictmodel:
        stop += as_dictmodel.get(key).numel()
        dictmodel.update(
            {
                key: flatmodel[start:stop]
                .view_as(as_dictmodel.get(key))
                .to(as_dictmodel.get(key).device)
            }
        )
        start = stop

    return collections.OrderedDict(dictmodel)


def model_weightload(
    dictmodel: collections.OrderedDict,
    instantiated_model,
    th_device=None,
    only_eval: bool = True,
):

    # Instantiate model as:
    # model = TheModelClass(*args, **kwargs)
    # Then pass model as instantiated_model

    instantiated_model.load_state_dict(dictmodel)

    # Enforce correct device usage
    if th_device is not None:
        if (th_device != "cpu") and (th_device != "cuda"):
            raise Exception(
                'model_weightload: th_device specified is neither "cpu" nor "cuda" (nor None).'
            )
        else:
            if th_device == "cuda" and not th.cuda.is_available():
                raise Exception(
                        'model_weightload: th_device specified as "cuda", but no CUDA device available.'
                )

            # Make sure to call input = input.to(device) on any input tensors that you feed to the model
            instantiated_model.to(th_device)

    if only_eval:
        instantiated_model.eval()

    return None
