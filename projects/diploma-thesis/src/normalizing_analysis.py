#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: normalizing_analysis.py
    Author: José Hilario
    Date created: 26.12.2018
    Date last modified: 21.05.2019
    Python Version: 3.5.6
'''

# Libraries
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.distributions import Normal
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import save, output_file
from bokeh.models import LinearAxis, Range1d
import numpy as np
# import foolbox

# Modules
from classes.Loader import Loader
import config

# Script arguments
dataset = sys.argv[1]

# Data loader
loader = Loader()
train_dataset = loader.load_train_dataset(dataset=config.ANALYSIS_DATASETS[dataset]['name'], path=config.ANALYSIS_DATASETS[dataset]['path'], **config.ANALYSIS_DATASETS[dataset]['config'])
test_dataset = loader.load_test_dataset(dataset=config.ANALYSIS_DATASETS[dataset]['name'], path=config.ANALYSIS_DATASETS[dataset]['path'], **config.ANALYSIS_DATASETS[dataset]['config'])
train = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
test = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
img_train, _ = iter(train).next()
img_test, _ = iter(test).next()
# numpy_images = np.vstack((img_train.numpy(), img_test))
numpy_images = img_train.numpy()

per_image_mean = np.mean(numpy_images, axis=(2,3)) #Shape (32,3)
per_image_std = np.std(numpy_images, axis=(2,3)) #Shape (32,3)

pop_channel_mean = np.mean(per_image_mean, axis=0) # Shape (3,)
pop_channel_std = np.mean(per_image_std, axis=0) # Shape (3,)

print('< mean={} >, < std={} >'.format(pop_channel_mean, pop_channel_std))
