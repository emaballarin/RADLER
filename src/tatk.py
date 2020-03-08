import torch as th

if __name__ == "__main__":
    import architectures as myarchs
    import weights_util as wutil
else:
    from src import architectures as myarchs
    from src import weights_util as wutil

replica = myarchs.AE_Decoder()

loaddict = th.load("bottleneck.pt")
wutil.model_weightload(loaddict, replica, "cuda")

faketensor = th.tensor([[0.0, 0.0]]).cuda()

print(wutil.dictmodel_flatten(th.load("mnist_cnn_small.pt"), "cuda"))
print(replica(faketensor))
