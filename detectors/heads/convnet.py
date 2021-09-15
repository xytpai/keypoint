import torch
import torch.nn as nn 
import torch.nn.functional as F 
from layers import *
import math


class ConvHead4(nn.Module):
    def __init__(self, channels, num_class):
        super().__init__()
        self.channels = channels
        self.num_class = num_class
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_class, kernel_size=3, padding=1))
        for block in [self.convs]:
            for layer in block.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.constant_(layer.bias, 0)
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)
        nn.init.constant_(self.convs[-1].bias, _bias)
    
    def forward(self, x):
        return self.convs(x)
