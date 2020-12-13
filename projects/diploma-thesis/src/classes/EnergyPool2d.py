#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: EnergyPool2d.py
    Author: José Hilario
    Date created: 05.04.2019
    Date last modified: 07.04.2019
    Python Version: 3.5.6
'''

import torch
import torch.nn as nn
from classes.ArgmaxAccPool2d import ArgmaxAccPool2d
from classes.ArgminAccPool2d import ArgminAccPool2d

class EnergyPool2d(nn.Module):
    """ Performs the operation Σ(1[i = argmax c] - 1[i = argmin c]) where c belongs to C.
        C is the set of all sliding window views obtained by the kernel.
        1[i = p] is the indicator function, adding a value of 1 at the position p.
        Intuitively, the operation counts the number of times there was a winner (argmax and argmin) at position i.
        argmax c represents a position winner of a max pooling operation, at a given sliding window view.
        argmin c represents a position winner of a min pooling operation, at a given sliding window view.
        The final result for each position is a count of max pooling wins, minus a count of min pooling wins.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`

    Example:
        >>> import torch
        >>> from classes.EnergyPool2d import EnergyPool2d
        >>> input = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [5, 6, 7, 8, 9, 1, 2, 3, 4],
                                  [9, 8, 7, 6, 5, 4, 3, 2, 1]],

                                 [[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [5, 6, 7, 8, 9, 1, 2, 3, 4],
                                  [9, 8, 7, 6, 5, 4, 3, 2, 1]]]])]
        >>> energy_pooling = EnergyPool2d(kernel_size=(2, 2), stride=(1, 1))
        >>> energy_pooling(input)
        tensor([[-1, -1, -1, -1,  0,  0,  1,  1,  1],
                [-1,  0,  1,  2,  4, -4, -2,  0,  1],
                [ 1,  1,  0, -1, -1,  1,  0,  0, -1]],

                [[-1, -1, -1, -1,  0,  0,  1,  1,  1],
                [-1,  0,  1,  2,  4, -4, -2,  0,  1],
                [ 1,  1,  0, -1, -1,  1,  0,  0, -1]]]]))
    """
    def __init__(self, kernel_size, stride=None):
        super(EnergyPool2d, self).__init__()
        self.max_winners = ArgmaxAccPool2d(kernel_size, stride)
        self.min_winners = ArgminAccPool2d(kernel_size, stride)

    def forward(self, input):
        return self.max_winners(input) - self.min_winners(input)
