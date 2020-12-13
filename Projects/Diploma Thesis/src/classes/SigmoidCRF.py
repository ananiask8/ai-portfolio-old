#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: SigmoidCRF.py
    Author: José Hilario
    Date created: 02.04.2019
    Date last modified: 09.04.2019
    Python Version: 3.5.6
'''

import torch
import torch.nn as nn
from classes.CliquePotentialsCRF import CliquePotentialsCRF

class SigmoidCRF(nn.Module):
    """ Stochastic activation given by the Gibbs distribution of a Conditional Random Field over a 2d lattice 

    Args:
        kernel_size: the size of the cliques to consider
        stride: the stride of the window; overlap of clique interactions
        alpha: the interaction parameter; strength of the interactions between nodes
        tol: accepted error for the calculation of potentials
        max_iter: maximum amount of iterations of the Frank-Wolfe algorithm
    """

    def __init__(self, kernel_size, stride, alpha=1, tol=1e-3, max_iter=100):
        super(SigmoidCRF, self).__init__()
        self.activations = nn.Sequential(
            CliquePotentialsCRF(kernel_size, stride, alpha=alpha, tol=tol, max_iter=max_iter),
            nn.Sigmoid()
        );

    def forward(self, x):
        return self.activations(x)
