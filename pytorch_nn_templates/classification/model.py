import utils

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision as tv


class FCModel(nn.Module):
    def __init__(self, n_in, n_out, n_hidden):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        
        self.n_hidden = n_hidden

        self.layers = nn.Sequential(
            nn.Linear(self.n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, self.n_out),
        )
        
    def forward(self, x):
        x = x.view(1, -1)
        return self.layers(x)


class CNNModel(nn.Module):
    def __init__(self, image_size, n_hidden, n_out, pool_size = 2, conv_size = 5, padding = 0, stride = 1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, n_hidden, conv_size, padding=padding, stride=stride),
            nn.MaxPool2d(pool_size, pool_size),
            nn.ReLU(),
            nn.Conv2d(n_hidden, n_hidden, conv_size, padding=padding, stride=stride),
            nn.MaxPool2d(pool_size, pool_size),
            nn.ReLU(),
        )
        

        image_size = int(utils.conv_calculator(image_size, conv_size, padding, stride)/2)
        image_size = int(utils.conv_calculator(image_size, conv_size, padding, stride)/2)

        self.lin = nn.Linear(n_hidden * image_size**2, n_out)


    def forward(self, x):
        x = self.conv(x)
        x = x.view(1, -1)
        x = self.lin(x)
        return x
