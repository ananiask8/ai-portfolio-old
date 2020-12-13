#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: test_adversarial.py
    Author: José Hilario
    Date created: 09.11.2018
    Date last modified: 21.05.2019
    Python Version: 3.5.6
'''

# Libraries
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bokeh.plotting import figure
from bokeh.io import save, output_file
import numpy as np

# Modules
from classes.Loader import Loader
import config

# Script arguments
model_name = sys.argv[1]
dataset = sys.argv[2]
net_type = sys.argv[3]
adv_name = sys.argv[4]

# Data loader
loader = Loader()
path = '/'.join([str(s).strip('/') for s in [config.ADV_STORE_PATH, adv_name]])
images, labels, distances = loader.load_adversarial_dataset(dataset=adv_name, path=path, preprocessing=config.ADV_DATASETS[dataset]['config']['preprocessing'])
training_type = 'Adv' if 'adv' in model_name else 'Std'

# ConvNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = config.network_model(net_type, dataset, 'test').to(device)
model.load_state_dict(torch.load(config.MODEL_STORE_PATH + model_name, map_location=device))
model.eval()
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    acc = correct / total
    print('Classification Accuracy for {} adversarial images: {} %'.format(len(images), acc * 100))

hist, edges = np.histogram(distances, density=False, bins='auto')
p = figure(x_range=(0, 1), y_range=(0, 150), title='({} training) {}: mu = {:0.4f}, std = {:0.4f}'.format(training_type, net_type, np.mean(distances), np.std(distances)))
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color='white')
output_file('{}/{}.html'.format(config.EXPERIMENTS_STORE_PATH, adv_name))
save(p)