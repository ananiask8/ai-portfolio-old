#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: _AccPool2d.py
    Author: José Hilario
    Date created: 31.03.2019
    Date last modified: 07.04.2019
    Python Version: 3.5.6
'''

import torch
import torch.nn as nn

class _AccPool2d(nn.Module):
    """ Accumulator max pooling base class module.
        Operations to accumulate the times each element in a 2d grid is selected as winner by a max pooling convolution.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`
    """

    def __init__(self, kernel_size, stride=None):
        super(_AccPool2d, self).__init__()
        self.winner_pool_2d = nn.MaxPool2d(kernel_size, stride, return_indices=True, ceil_mode=False)
        self.flattening_offset = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_flattening_offset(self, shape):
        N, C, H, W = shape
        if self.flattening_offset is not None:
            N_offset, _, _ = self.flattening_offset.size() 

        if self.flattening_offset is None or N != N_offset:
            self.flattening_offset = torch.tensor([[i * H * W] for i in range(N*C)])
            self.flattening_offset = self.flattening_offset.type(torch.LongTensor).to(self.device).view(N, C, -1)

        return self.flattening_offset

    def get_filled_tensor(self, shape, val, cast):
        types = {
            'LongTensor': torch.LongTensor,
            'DoubleTensor': torch.DoubleTensor,
            'FloatTensor': torch.FloatTensor
        }
        return types[cast](shape).to(self.device).fill_(val)

    def accumulate_at_winner_position(self, argmax_indices, input_shape):
        N, C, _, _ = input_shape
        argmax_indices = argmax_indices.view(N, C, -1)
        argmax_indices += self.get_flattening_offset(input_shape)
        __0__ = self.get_filled_tensor(input_shape, 0, 'LongTensor')
        __1__ = self.get_filled_tensor(argmax_indices.size(), 1, 'LongTensor')
        return __0__.put_(argmax_indices.view(-1), __1__, accumulate=True)
