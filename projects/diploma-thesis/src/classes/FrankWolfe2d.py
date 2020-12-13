#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: FrankWolfe2d.py
    Author: José Hilario
    Date created: 05.04.2019
    Date last modified: 21.05.2019
    Python Version: 3.5.6
'''

import torch
from torch.autograd.function import Function

class FrankWolfe2d(Function):
    """ Implementation of the Frank-Wolfe algorithm for functions defined over 2d grids
        Solves the problem: min f(x) subject to x in D
        D is a compact convex set in a vector space
        f: D → ℝ is a convex differentiable function

    Args:
        x: the 4d input to the algorithm
        gradient: the gradient of the function f
        min_first_order_taylor_approx: closed form minimization of the first order Taylor approximation of f around x
        lr: learning rate as a function of the iteration number
        tol: stopping criteria with respect to error tolerance
        max_iter: stopping criteria with respect to iterations of the algorithm
    """

    @staticmethod
    def forward(ctx, x, gradient, min_first_order_taylor_approx, tol=1e-3, max_iter=200, lr=lambda t, **kwargs: 2.0 / (t + 2.0)):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_shape = x.size()
        N, C, H, W = x_shape
        g, indices = torch.FloatTensor(x_shape).to(device), torch.ones(x_shape).type(torch.ByteTensor).to(device)
        for t in range(max_iter):
            r = gradient(x)
            s = min_first_order_taylor_approx(r)
            g = torch.einsum('nchw,nchw->nc', [x - s, r])[:,:,None,None].expand(N, C, H, W)
            if (g < tol).all():
                break
            gamma = lr(t, x=x, direction=s)
            x_t = x
            x = ((1 - gamma)*x_t + gamma*s)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
