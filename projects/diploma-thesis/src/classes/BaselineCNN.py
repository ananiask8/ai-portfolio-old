#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: BaselineCNN.py
    Author: José Hilario
    Date created: 08.11.2018
    Date last modified: 21.05.2019
    Python Version: 3.5.6
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.distributions import Normal
import torchvision.datasets
from classes.GaussianNoise import GaussianNoise

class BaselineCNN(nn.Module):
    # # # # Wout = (Win - F + 2P)/S + 1
    # Win = 28
    # (28 - 5 + 0) / 1 + 1 = 24
    # (24 - 3 + 0) / 3 + 1 = 8
    # (8 - 5 + 0) / 1 + 1 = 4
    # (4 - 2 + 0) / 2 + 1 = 2
    # Wout = 1
    def __init__(self, c=1, n=10, std=0.):
        super(BaselineCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, n, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(n),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.log_softmax(out)
        return out
