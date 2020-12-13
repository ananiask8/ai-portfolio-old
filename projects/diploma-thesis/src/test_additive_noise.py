#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: test_additive_noise.py
    Author: José Hilario
    Date created: 09.11.2018
    Date last modified: 21.05.2019
    Python Version: 3.5.6
'''

# Libraries
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions import Normal
from bokeh.plotting import figure
from bokeh.io import save, output_file
from bokeh.palettes import Dark2_5 as palette
import numpy as np
import os

# Modules
from classes.Loader import Loader
import config

# Script arguments
model_names = sys.argv[1].split(',')
net_types = sys.argv[2].split(',')
dataset = sys.argv[3]
output_filename = sys.argv[4]

# Data loader
loader = Loader()
test_dataset = loader.load_test_dataset(dataset=config.DATASETS[dataset]['name'], path=config.DATASETS[dataset]['path'], **config.DATASETS[dataset]['config'])
test_loader = DataLoader(dataset=test_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=False)

# ConvNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}
for model_name, net_type in zip(model_names, net_types):
    model = config.network_model(net_type, dataset, 'test').to(device)
    model.load_state_dict(torch.load(config.MODEL_STORE_PATH + model_name, map_location=device))
    model.eval()
    models[model_name] = model

# Initialize distributions
normals = []
for std in [torch.tensor(0.0), *torch.logspace(start=-1.5, end=0., steps=20)]:
    N = {'std': std, 'dist': Normal(torch.tensor(0.0), std / config.STATISTICS[dataset]['STD'])}
    if torch.cuda.is_available():
        N['dist'].loc = N['dist'].loc.cuda()
        N['dist'].scale = N['dist'].scale.cuda()
    normals.append(N)

def print_statistics(tp, fp, tn, fn, vanilla_correct, total):
    vanilla_acc = vanilla_correct / total
    accepted = tp + fp
    rejected = total - accepted
    rejected_ratio =  rejected / total
    safe_correct = (tp / accepted) if accepted > 0 else float('inf')
    fpr = fp / (fp + tp)
    fnr = fn / (fn + tn)
    print('Vanilla Accuracy: {:0.2f} %'.format(vanilla_acc * 100))
    print('Safe Accuracy (True Positive Rate): {:0.2f} %'.format(safe_correct * 100))
    print('False Positive Rate: {:0.2f} %'.format(fpr * 100))
    print('False Negative Rate: {:0.2f} %'.format(fnr * 100))
    print('Rejected: {:0.2f} %'.format(rejected_ratio * 100))
    return safe_correct, fpr, fnr, rejected_ratio

# Test the models for validation set with additive noise
acc = {}
safe_acc = {}
accepted = {}
for model_name, model in models.items():
    acc[model_name] = np.array([])
    safe_acc[model_name] = np.array([])
    accepted[model_name] = np.array([])
    print('\r\nModel: {}'.format(model_name))
    for normal in normals:
        std = normal['std']
        model.eval()
        print('\r\nT = {{x + d | d ~ N(0, {:0.4f}), x in D}}. |T| = {}'.format(std, len(test_dataset)))
        with torch.no_grad():
            correct = 0
            total = 0
            tp, fp, tn, fn = 0, 0, 0, 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                images = images + normal['dist'].sample(images.size())
                images = torch.clamp(images, min=config.STATISTICS[dataset]['MIN'], max=config.STATISTICS[dataset]['MAX'])
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                top_correct = torch.exp(outputs[predicted == labels]).topk(2)[0]
                top_incorrect = torch.exp(outputs[predicted != labels]).topk(2)[0]
                tp += (top_correct[:,0] >= config.CONFIDENCE_THRESHOLD).sum().item()
                fn += (top_correct[:,0] < config.CONFIDENCE_THRESHOLD).sum().item()
                fp += (top_incorrect[:,0] >= config.CONFIDENCE_THRESHOLD).sum().item()
                tn += (top_incorrect[:,0] < config.CONFIDENCE_THRESHOLD).sum().item()
            t_acc = correct / total
            safe_correct, fpr, fnr, rejected_ratio = print_statistics(tp, fp, tn, fn, correct, total)
            acc[model_name] = np.append(acc[model_name], [std, t_acc])
            safe_acc[model_name] = np.append(safe_acc[model_name], [std, safe_correct])
            accepted[model_name] = np.append(accepted[model_name], [std, 1 - rejected_ratio])
        acc[model_name] = np.reshape(acc[model_name], (-1,2))
        safe_acc[model_name] = np.reshape(safe_acc[model_name], (-1,2))
        accepted[model_name] = np.reshape(accepted[model_name], (-1,2))

    acc[model_name] = np.reshape(acc[model_name], (-1,2))

# Plot
path = '{}/{}/'.format(config.EXPERIMENTS_STORE_PATH.rstrip('/'), output_filename)
if not os.path.isdir(path):
    os.makedirs(path)

p = figure(x_axis_label='Gaussian Noise N(0, σ)', x_axis_type="log", y_axis_label='Accuracy', width=850, y_range=(0, 1), title='Effects of Additive Noise at the Input')
for model_name, color in zip(models.keys(), palette):
    p.line(np.array(acc[model_name][:,0]), np.array(acc[model_name][:,1]), color=color, legend=model_name)
output_file('{}/{}/acc.html'.format(config.EXPERIMENTS_STORE_PATH.rstrip('/'), output_filename))
save(p)

p = figure(x_axis_label='Gaussian Noise N(0, σ)', x_axis_type="log", y_axis_label='Safe Accuracy (p > {})'.format(config.CONFIDENCE_THRESHOLD), width=850, y_range=(0, 1), title='Effects of Additive Noise at the Input')
for model_name, color in zip(models.keys(), palette):
    p.line(np.array(safe_acc[model_name][:,0]), np.array(safe_acc[model_name][:,1]), color=color, legend=model_name)
output_file('{}/{}/safe_acc.html'.format(config.EXPERIMENTS_STORE_PATH.rstrip('/'), output_filename))
save(p)

p = figure(x_axis_label='Gaussian Noise N(0, σ)', x_axis_type="log", y_axis_label='Accepted Ratio (p > {})'.format(config.CONFIDENCE_THRESHOLD), width=850, y_range=(0, 1), title='Effects of Additive Noise at the Input')
for model_name, color in zip(models.keys(), palette):
    p.line(np.array(accepted[model_name][:,0]), np.array(accepted[model_name][:,1]), color=color, legend=model_name)
output_file('{}/{}/accepted.html'.format(config.EXPERIMENTS_STORE_PATH.rstrip('/'), output_filename))
save(p)
