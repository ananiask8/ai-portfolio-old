#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: create_feature_images.py
    Author: José Hilario
    Date created: 20.04.2019
    Date last modified: 21.05.2019
    Python Version: 3.5.6
'''

import torch
from classes.ArgmaxAccPool2d import ArgmaxAccPool2d
from classes.ArgminAccPool2d import ArgminAccPool2d
from classes.EnergyPool2d import EnergyPool2d
from classes.FrankWolfe2d import FrankWolfe2d
import fixtures as td

class Test:
  def __init__(self, id, scenario, input, expect, tol=1e-5):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.id = id
    self.scenario = scenario
    self.input = [x.to(self.device) if hasattr(x, 'to') else x for x in input]
    self.expect = expect.to(self.device)
    self.tol = tol

  def all_eq(self, f):
    out = f(*self.input)
    print('====================================================================================')
    print('Test {} – {}: {}'.format(self.id, self.scenario, 'passing' if torch.allclose(out, self.expect, atol=self.tol) else 'failing'))
    if not torch.allclose(out, self.expect, atol=self.tol):
      print('Expected: {}'.format(self.expect))
      print('Got: {}'.format(out))
  def norm_eq(self, f):
    out = f(*self.input)
    err = torch.norm(out - self.expect, float('inf'))
    print('====================================================================================')
    print('Test {} – {}: {}'.format(self.id, self.scenario, 'passing' if err <= self.tol else 'failing'))
    if err > self.tol:
      print('Expected: {}'.format(self.expect))
      print('Got: {}'.format(out))
      print('L_inf of error: {}'.format(err))


test_1 = Test('#1', 'ArgmaxAccPool2d', td.in_1, td.expect_1)
maxacc2d = ArgmaxAccPool2d(kernel_size=(2, 2), stride=(1, 1))
test_1.all_eq(maxacc2d)

test_2 = Test('#2', 'ArgminAccPool2d', td.in_2, td.expect_2)
minacc2d = ArgminAccPool2d(kernel_size=(2, 2), stride=(1, 1))
test_2.all_eq(minacc2d)

test_3 = Test('#3', 'EnergyPool2d', td.in_3, td.expect_3)
epooling2d = EnergyPool2d(kernel_size=(2, 2), stride=(1, 1))
test_3.all_eq(epooling2d)

test_4 = Test('#4', 'ArgmaxAccPool2d', td.in_4, td.expect_4)
maxacc2d = ArgmaxAccPool2d(kernel_size=(2, 2), stride=(1, 1))
test_4.all_eq(maxacc2d)

test_5 = Test('#5', 'FrankWolfe2d: f(x) = (x1 - 2)^2 + (x2 - 2)^2, D in [-2, 4]', td.in_5, td.expect_5, tol=0.05)
test_5.norm_eq(FrankWolfe2d.apply)

test_6 = Test('#6', 'FrankWolfe2d: f(x) = (x1 + 2)^2 + (x2 + 2)^2, D in [-4, -2]', td.in_6, td.expect_6, tol=0.05)
test_6.norm_eq(FrankWolfe2d.apply)

test_7 = Test('#7', 'FrankWolfe2d: f(x) = (x − 0.5)^2 + 2x, D in [-1, 2]', td.in_7, td.expect_7, tol=0.05)
test_7.norm_eq(FrankWolfe2d.apply)

test_8 = Test('#8', 'FrankWolfe2d with default lr: f(x) = (x − 0.5)^2 + 2x, D in [-1, 2]', td.in_8, td.expect_8, tol=0.05)
test_8.norm_eq(FrankWolfe2d.apply)

test_9 = Test('#9', 'FrankWolfe2d with default lr: f(x) = (x1 − 0.5)^2 + 2x1 + (x2 + 2)^2, D in [-10, 2]', td.in_9, td.expect_9, tol=0.05)
test_9.norm_eq(FrankWolfe2d.apply)
