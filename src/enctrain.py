from __future__ import print_function
import argparse
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()

        # TODO: Batch-Normalize some layers as to prevent weight explosion!
        # TODO: e.g. fully connected layers: (1), 2, 3, 4, (5). () = maybe

        # Layer shapes
        inputsize: int = 2
        hiddensize1: int = 100
        hiddensize2: int = 100
        hiddensize3: int = 100
        hiddensize4: int = 100
        hiddensize5: int = 100
        outputsize: int = 431080    # MNIST-like

        # Layer types
        self.fc1 = nn.Linear(inputsize, hiddensize1)
        self.fc2 = nn.Linear(hiddensize1, hiddensize2)
        self.fc3 = nn.Linear(hiddensize2, hiddensize3)
        self.fc4 = nn.Linear(hiddensize3, hiddensize4)
        self.fc5 = nn.Linear(hiddensize4, hiddensize5)
        self.fc6 = nn.Linear(hiddensize5, outputsize)

    def forward(self, x):

        # Connectivity ops
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        # Output
        return self.f6(x)   # Last transform is a linear transform
