import torch as th
import collections


def modeldict_numel(modeldict: collections.OrderedDict):
    elems: int = 0
    for key in modeldict:
        elems += modeldict.get(key).numel()
    return elems


def model_flatten(dictmodel: collections.OrderedDict, th_device="cuda"):

    if (th_device != "cpu") and (th_device != "cuda"):
        raise Exception('model_flatten: th_device specified is neither "cpu" nor "cuda".')

    flatmodel: th.Tensor = th.tensor([]).to(th_device)

    for key in dictmodel:
        to_add: th.Tensor = dictmodel.get(key).flatten().to(th_device)
        flatmodel = th.cat((flatmodel, to_add), dim=0)

    flatmodel = flatmodel.to(th_device)

    # Cautionary consistency check
    if flatmodel.size() != th.Size([modeldict_numel(dictmodel)]):
        print("model_flatten: Number-of-Elements mismatch between FLAT and DICT models!")
        raise Exception('THIS EXCEPTION IS A DIAGNOSTIC FEATURE. SHOULD NEVER HAPPEN! See printed message for errors.')

    return flatmodel


def model_unflatten(flatmodel: th.Tensor, as_dictmodel: collections.OrderedDict):

    # Consistency check
    if flatmodel.size() != th.Size([modeldict_numel(as_dictmodel)]):
        raise Exception('model_unflatten: Number-of-Elements mismatch between FLAT and AS_DICT models!')

    modeldict: dict = {}
    start: int = 0
    stop: int = 0

    for key in as_dictmodel:
        stop += as_dictmodel.get(key).numel()
        modeldict.update({key: flatmodel[start:stop].view_as(as_dictmodel.get(key)).to(as_dictmodel.get(key).device)})
        start = stop

    return collections.OrderedDict(modeldict)


def model_weightload(modeldict: collections.OrderedDict, instantiated_model, th_device=None, only_eval: bool = True):

    # Instantiate model as
    # model = TheModelClass(*args, **kwargs)
    # Then pass model as instantiated_model

    instantiated_model.load_state_dict(modeldict)

    if th_device is not None:
        if (th_device != "cpu") and (th_device != "cuda"):
            raise Exception('model_weightload: th_device specified is neither "cpu" nor "cuda" (nor None).')
        else:
            instantiated_model.to(th_device)
            # Make sure to call input = input.to(device) on any input tensors that you feed to the model

    if only_eval:
        instantiated_model.eval()

    return None
