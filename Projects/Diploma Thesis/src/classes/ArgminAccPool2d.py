#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: ArgminAccPool2d.py
    Author: José Hilario
    Date created: 05.04.2019
    Date last modified: 07.04.2019
    Python Version: 3.5.6
'''

from classes._AccPool2d import _AccPool2d

class ArgminAccPool2d(_AccPool2d):
    """ Counts the amount of times each element in a 2d grid was the winner of a min pooling operation.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`

    Example:
        >>> import torch
        >>> from classes.ArgminAccPool2d import ArgminAccPool2d
        >>> input = torch.Tensor([[[
              [1, 2, 3, 4, 5, 6, 7, 8, 9],
              [5, 6, 7, 8, 9, 1, 2, 3, 4],
              [9, 8, 7, 6, 5, 4, 3, 2, 1]]]])
        >>> c = ArgminAccPool2d(kernel_size=(2, 2), stride=(1, 1))
        >>> c(input)
        tensor([[[[1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 4, 2, 1, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 1]]]])
    """
    def __init__(self, *args, **kwargs):
        super(ArgminAccPool2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        _, indices = self.winner_pool_2d(-input)
        return self.accumulate_at_winner_position(indices, input.size())
