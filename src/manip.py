import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

if __name__ == "__main__":
    import radler_util as rutil
else:
    from src import radler_util as rutil


# Chollet-Chintala-Ansuini architecture, ensmalled ;)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 1)
        self.conv2 = nn.Conv2d(8, 16, 5, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# DECODER (Scaffold):
class AE_Decoder(nn.Module):
    def __init__(self, code_size=2, data_size=20522):
        super(AE_Decoder, self).__init__()

        self.code_size = code_size
        self.data_size = data_size

        self.fc1 = nn.Linear(self.code_size, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, self.data_size)

    def forward(self, x):
        # Boilerplate
        # batch_size, channels, width, height = x.sizes()
        # x = x.view(batch_size, -1)

        # Layer 1
        x = self.fc1(x)
        x = F.relu(x)

        # Layer 2
        x = self.fc2(x)
        x = F.relu(x)

        # Layer 3
        x = self.fc3(x)
        x = F.relu(x)

        # Layer 4
        x = self.fc4(x)

        # Generated output
        code = th.tanh(x)  # We want our data to be "probably also negative"

        return code


loaddict1 = th.load("mnist_cnn_small.pt")
pippo1 = rutil.dictmodel_flatten(loaddict1)
print(pippo1)

loaddict2 = th.load("bottleneck.pt")
mymodel2 = AE_Decoder()
rutil.model_weightload(loaddict2, mymodel2, "cuda")
pippo2 = mymodel2(th.tensor([0.0, 0.0]).cuda())

print(pippo1)
print(pippo2)
print("\n\n\n")
print((th.abs(pippo1) - th.abs(pippo2)).sum())
