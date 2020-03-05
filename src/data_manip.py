import torch as th
import torchsummary

from src import radler_util as rutil

loaddict = th.load("mnist_cnn_small.pt")

vector = rutil.dictmodel_flatten(loaddict)

pippo = vector.clone().detach().repeat(10, 5, 1)

print(pippo[5].size())
