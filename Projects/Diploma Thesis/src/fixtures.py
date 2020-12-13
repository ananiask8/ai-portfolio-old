#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: fixtures.py
    Author: José Hilario
    Date created: 20.04.2019
    Date last modified: 21.05.2019
    Python Version: 3.5.6
'''

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TEST #01
in_1 = [torch.Tensor([[
 [[1, 2, 3, 4, 5, 6, 7, 8, 9],
  [5, 6, 7, 8, 9, 1, 2, 3, 4],
  [9, 8, 7, 6, 5, 4, 3, 2, 1]]]])]
expect_1 = torch.LongTensor([[
 [[0, 0, 0, 0, 0, 0, 1, 1, 1],
  [0, 1, 1, 2, 4, 0, 0, 1, 1],
  [1, 1, 0, 0, 0, 1, 0, 0, 0]]]])

# TEST #02
in_2 = [torch.Tensor([[
 [[1, 2, 3, 4, 5, 6, 7, 8, 9],
  [5, 6, 7, 8, 9, 1, 2, 3, 4],
  [9, 8, 7, 6, 5, 4, 3, 2, 1]]]])]
expect_2 = torch.LongTensor([[
 [[1, 1, 1, 1, 0, 0, 0, 0, 0],
  [1, 1, 0, 0, 0, 4, 2, 1, 0],
  [0, 0, 0, 1, 1, 0, 0, 0, 1]]]])

# TEST #03
in_3 = [torch.Tensor([[
 [[1, 2, 3, 4, 5, 6, 7, 8, 9],
  [5, 6, 7, 8, 9, 1, 2, 3, 4],
  [9, 8, 7, 6, 5, 4, 3, 2, 1]],

 [[1, 2, 3, 4, 5, 6, 7, 8, 9],
  [5, 6, 7, 8, 9, 1, 2, 3, 4],
  [9, 8, 7, 6, 5, 4, 3, 2, 1]]]])]
expect_3 = torch.LongTensor([[
 [[-1, -1, -1, -1,  0,  0,  1,  1,  1],
  [-1,  0,  1,  2,  4, -4, -2,  0,  1],
  [ 1,  1,  0, -1, -1,  1,  0,  0, -1]],

 [[-1, -1, -1, -1,  0,  0,  1,  1,  1],
  [-1,  0,  1,  2,  4, -4, -2,  0,  1],
  [ 1,  1,  0, -1, -1,  1,  0,  0, -1]]]])

# TEST #04
in_4 = [torch.Tensor([
  [[[1, 2, 1],
    [2, 3, 2],
    [1, 2, 1]],
   [[3, 2, 3],
    [2, 1, 2],
    [3, 2, 3]],
   [[2, 3, 2],
    [3, 1, 3],
    [2, 3, 2]]],
  [[[3, 2, 3],
    [2, 1, 2],
    [3, 2, 3]],
   [[1, 2, 1],
    [2, 3, 2],
    [1, 2, 1]],
   [[1, 3, 1],
    [3, 2, 3],
    [1, 3, 1]]],
  [[[1, 3, 1],
    [3, 2, 3],
    [1, 3, 1]],
   [[3, 2, 3],
    [2, 1, 2],
    [3, 2, 3]],
   [[1, 2, 1],
    [2, 3, 2],
    [1, 2, 1]]]])]
expect_4 = torch.LongTensor([
  [[[0, 0, 0],
    [0, 4, 0],
    [0, 0, 0]],
   [[1, 0, 1],
    [0, 0, 0],
    [1, 0, 1]],
   [[0, 2, 0],
    [1, 0, 1],
    [0, 0, 0]]],
  [[[1, 0, 1],
    [0, 0, 0],
    [1, 0, 1]],
   [[0, 0, 0],
    [0, 4, 0],
    [0, 0, 0]],
   [[0, 2, 0],
    [1, 0, 1],
    [0, 0, 0]]],
  [[[0, 2, 0],
    [1, 0, 1],
    [0, 0, 0]],
   [[1, 0, 1],
    [0, 0, 0],
    [1, 0, 1]],
   [[0, 0, 0],
    [0, 4, 0],
    [0, 0, 0]]]])

# TEST #05
shape_5 = torch.Size([1, 1, 1, 2])
domain_5, init_5 = [-2, 4], 4
v = torch.FloatTensor(shape_5).fill_(init_5).to(device)
def gradient(x):
  return 2*(x - 2)
def lr(t, **kwargs):
  return 2.0 / (t + 2.0)
def min_first_order_taylor_approx(grad_fx):
  s = torch.FloatTensor(grad_fx.size()).to(device).fill_(0)
  s[grad_fx <= 0] = domain_5[1]
  s[grad_fx > 0] = domain_5[0]
  return s
in_5 = [v, gradient, min_first_order_taylor_approx, 1e-3, 100, lr]
expect_5 = torch.Tensor([[[2, 2]]])

# TEST #06
shape_6 = torch.Size([1, 1, 1, 2])
domain_6, init_6 = [-4, -2], -4
v = torch.FloatTensor(shape_6).fill_(init_6).to(device)
def gradient(x):
  return 2*(x + 2)
def lr(t, **kwargs):
  return 0.95 ** t
def min_first_order_taylor_approx(grad_fx):
  s = torch.FloatTensor(grad_fx.size()).to(device).fill_(0)
  s[grad_fx <= 0] = domain_6[1]
  s[grad_fx > 0] = domain_6[0]
  return s
in_6 = [v, gradient, min_first_order_taylor_approx, 1e-3, 100, lr]
expect_6 = torch.Tensor([-2])

# TEST #07
shape_7 = torch.Size([1, 1, 1, 2])
domain_7, init_7 = [-1, 2], 2
v = torch.FloatTensor(shape_7).fill_(init_7).to(device)
def gradient(x):
  return 2*(x - 0.5) + 2
def lr(t, **kwargs):
  return 0.85 ** t
def min_first_order_taylor_approx(grad_fx):
  s = torch.FloatTensor(grad_fx.size()).to(device).fill_(0)
  s[grad_fx <= 0] = domain_7[1]
  s[grad_fx > 0] = domain_7[0]
  return s
in_7 = [v, gradient, min_first_order_taylor_approx, 1e-3, 100, lr]
expect_7 = torch.Tensor([-0.5])

# TEST #08
shape_8 = torch.Size([2, 2, 2, 2])
domain_8, init_8 = [-1, 2], 2
v = torch.FloatTensor(shape_8).fill_(init_8).to(device)
def gradient(x):
  return 2*(x - 0.5) + 2
def lr(t, **kwargs):
  return 2.0 / (t + 2.0)
def min_first_order_taylor_approx(grad_fx):
  s = torch.FloatTensor(grad_fx.size()).to(device).fill_(0)
  s[grad_fx <= 0] = domain_8[1]
  s[grad_fx > 0] = domain_8[0]
  return s
in_8 = [v, gradient, min_first_order_taylor_approx, 1e-3, 100, lr]
expect_8 = torch.Tensor([-0.5])
print(expect_8.size())

# TEST #09
shape_9 = torch.Size([2, 1, 1, 2])
domain_9, init_9 = [-10, 2], 2
v = torch.FloatTensor(shape_9).fill_(init_9).to(device)
v[0,:,:,:] = -1
v[1,:,:,:] = 2
def gradient(x):
  dx = torch.FloatTensor(x.size()).to(device)
  dx[:,:,:,0] = 2*(x[:,:,:,0] - 0.5) + 2
  dx[:,:,:,1] = 2*(x[:,:,:,1] + 2)
  return dx
def lr(t, **kwargs):
  return 2.0 / (t + 2.0)
def min_first_order_taylor_approx(grad_fx):
  s = torch.FloatTensor(grad_fx.size()).to(device).fill_(0)
  s[grad_fx <= 0] = domain_9[1]
  s[grad_fx > 0] = domain_9[0]
  return s
in_9 = [v, gradient, min_first_order_taylor_approx]
expect_9 = torch.Tensor([-0.5, -2])
        