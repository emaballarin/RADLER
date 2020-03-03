import torch as th
import torchsummary

from src import radler_util as rutil
from src import mnist_baseline as mymnist

loaddict = th.load("mnist_cnn.pt")

tryclass = mymnist.Net()
rutil.model_weightload(loaddict, tryclass, "cuda")

print(rutil.dictmodel_numel(loaddict))

# print(loaddict)

# torchsummary.summary(tryclass, (1, 28, 28))

# pippo = ansumnist.Net()
# torchsummary.summary(pippo, (1,28,28))
