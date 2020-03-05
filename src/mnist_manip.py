import torch as th
import torchsummary

from src import radler_util as rutil
from src import mnist_baseline as mnistbase
from src import mnist_small as mnistsmall

mnistclass = mnistsmall

loaddict = th.load("mnist_cnn_small.pt")

tryclass = mnistclass.Net()
rutil.model_weightload(loaddict, tryclass, "cuda")

print(rutil.dictmodel_numel(loaddict))

print(rutil.dictmodel_flatten(loaddict))

# torchsummary.summary(tryclass, (1, 28, 28))
