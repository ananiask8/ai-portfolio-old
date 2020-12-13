#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: PGDAttack.py
    Author: José Hilario
    Date created: 30.03.2019
    Date last modified: 15.04.2019
    Python Version: 3.5.6
'''

import torch
import torch.nn as nn
from torch.autograd.function import Function

class PGDAttack(nn.Module):
    def __init__(self, model, loss, config):
        super(PGDAttack, self).__init__()
        self.model = model
        self.loss = loss
        self.k = config['k']
        self.epsilon = config['epsilon']
        self.step_size = config['step_size']
        self.random_start = config['random_start']
        self.range = config['range']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, targets):
        x = inputs.detach()
        if self.random_start:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = self.loss(logits, targets)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, *self.range)

        return x
