import torch as th
import collections


def modeldict_numel(modeldict: collections.OrderedDict):
    elems: int = 0
    for key in modeldict:
        elems += modeldict.get(key).numel()
    return elems


def model_unflatten(flatmodel: th.Tensor, as_dictmodel: collections.OrderedDict):

    # Consistency check
    if flatmodel.size() != th.Size([modeldict_numel(as_dictmodel)]):
        raise Exception('model_unflatten: Number-of-Elements mismatch between FLAT and AS_DICT models!')

    modeldict: dict = {}
    start: int = 0
    stop: int = 0

    for key in as_dictmodel:
        stop += as_dictmodel.get(key).numel()
        modeldict.update({key: flatmodel[start:stop].view_as(as_dictmodel.get(key)).clone().to(as_dictmodel.get(key)
                                                                                               .device)})
        start = stop

    return collections.OrderedDict(modeldict)

