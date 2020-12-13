#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: CliquePotentialsCRF.py
    Author: José Hilario
    Date created: 05.04.2019
    Date last modified: 04.05.2019
    Python Version: 3.5.6
'''

import torch
import torch.nn as nn
from torch.autograd.function import Function
from classes.EnergyPool2d import EnergyPool2d
from classes.FrankWolfe2d import FrankWolfe2d

class CliquePotentialsCRF(nn.Module):
    """ Considering a Conditional Random Field on a 2d grid
        This module obtains the potentials given by the interactions between the nodes of the cliques

    Args:
        kernel_size: the size of the cliques to consider
        stride: the stride of the window; overlap of clique interactions
        alpha: the interaction parameter; strength of the interactions between nodes
        tol: accepted error for the calculation of potentials
        max_iter: maximum amount of iterations of the Frank-Wolfe algorithm
    """

    def __init__(self, kernel_size, stride, alpha=1, tol=1e-3, max_iter=100):
        super(CliquePotentialsCRF, self).__init__()
        self.energy_pooling = EnergyPool2d(kernel_size, stride)
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def device_is_cuda(self):
        return self.device.type == 'cuda'

    def gradient_wrapper(self, beta):
        def gradient(v):
            return (torch.exp(v + beta)) / (torch.exp(v + beta) + 1)
        return gradient

    def min_first_order_taylor_approx(self, w):
        out = -self.alpha * self.energy_pooling(w)
        return out.type((torch.cuda if self.device_is_cuda() else torch).FloatTensor)

    def forward(self, beta):
        # Original problem
        # v_init = beta.clone()
        # bias = torch.cuda.FloatTensor(beta.shape).fill_(0) if torch.cuda.is_available() else torch.zeros(beta.shape) #transformed
        # return -FrankWolfe2d.apply(v_init, self.gradient_of_objective(bias), self.min_first_order_taylor_approx, self.tol, self.max_iter)

        # Transformed problem
        v_init = torch.cuda.FloatTensor(beta.shape).fill_(0) if torch.cuda.is_available() else torch.zeros(beta.shape) #transformed
        return -(beta + FrankWolfe2d.apply(v_init, self.gradient_wrapper(beta), self.min_first_order_taylor_approx, self.tol, self.max_iter))
