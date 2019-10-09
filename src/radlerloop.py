import argparse

import jax.numpy as np
import numpy as onp

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

import torchx as thx

import art

# Stub.

# TODO:
# Autoencode SoTA weights + GAN train generator over mixed metric: MINIMIZE (alpha)*(error) + (beta)*(unrobustness)
# GAN train discriminator over MAXIMIXE (beta)*(unrobustness), given input to model and full flattened weight vector
# UN-robustness metric: sample over model input space, attack, count successful attacks -> UNROB = successful/total
