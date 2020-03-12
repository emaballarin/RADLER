import torch as th

if __name__ == "__main__":
    import architectures as myarchs
    import losses as mylosses
    import weights_util as wutil

else:
    from src import architectures as myarchs
    from src import losses as mylosses
    from src import weights_util as wutil

# TESTS:
loaddict = th.load("bottleneck.pt")
pippo = myarchs.AE_Decoder()
wutil.model_weightload(loaddict, pippo, "cuda")

faketensor = th.tensor((0.1, 0.2))
wout = pippo(faketensor.cuda())

print(mylosses.mnistnet_earloss_wfs(0.5, wout, 250, "cuda"))
