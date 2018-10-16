import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision as tv
import torch.optim as optim


class FCModel(nn.Module):
    def __init__(self, hyperparams):
        super(FCModel, self).__init__()

        self.in_neurons, self.out_neurons, _ = hyperparams

        self.fc1 = nn.Linear(self.in_neurons, 2 * self.in_neurons)
        self.fc2 = nn.Linear(2 * self.in_neurons, 2 * self.in_neurons)
        self.fc3 = nn.Linear(2 * self.in_neurons, self.out_neurons)
    def forward(self, x):
        x = x.view(-1, self.in_neurons)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

class CNNModel(nn.Module):
    def __init__(self, hyperparams):
        super(CNNModel, self).__init__()

        self.in_neurons, self.out_neurons, self.k_size = hyperparams

        self.conv1 = nn.Conv2d(self.in_neurons, 2 * self.in_neurons, self.k_size)
        self.conv2 = nn.Conv2d(2 * self.in_neurons, 2 * self.in_neurons, self.k_size)
        self.conv3 = nn.Conv2d(2 * self.in_neurons, self.out_neurons, self.k_size)
    def forward(self, x):
        #x = x.view(-1, in_neurons)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x.view(-1, self.out_neurons)
