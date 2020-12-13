#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: train.py
    Author: José Hilario
    Date created: 08.11.2018
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

# Modules
from classes.Loader import Loader
from classes.PGDAttack import PGDAttack
import config

# Script arguments
model_name = sys.argv[1]
dataset = sys.argv[2]
net_type = sys.argv[3]
training_type = sys.argv[4]

# Data loader
loader = Loader()
train_dataset = loader.load_train_dataset(dataset=config.DATASETS[dataset]['name'], path=config.DATASETS[dataset]['path'], **config.DATASETS[dataset]['config'])
test_dataset = loader.load_test_dataset(dataset=config.DATASETS[dataset]['name'], path=config.DATASETS[dataset]['path'], **config.DATASETS[dataset]['config'])
dataloaders = {
    'train': DataLoader(dataset=train_dataset, batch_size=config.TRAINING_BATCH_SIZE, shuffle=True),
    'val': DataLoader(dataset=test_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=False)
}

# ConvNet
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = config.network_model(net_type, dataset, 'train').to(device)
try:
    model.load_state_dict(torch.load(config.MODEL_STORE_PATH + model_name, map_location=device))
except: pass
criterion = nn.NLLLoss()
attack = PGDAttack(model, criterion, config.ADV_TRAINING_SETUP[dataset])
optimizer = torch.optim.SGD(model.parameters(), lr=config.OPTIMIZER['lr'], momentum=config.OPTIMIZER['mt'], weight_decay=config.OPTIMIZER['wd'])

# Train the model
loss_list = {'train':[], 'val':[]}
acc_list = {'train':[], 'val':[]}
loss_means = []
acc_means = []

iters_per_epoch = len(dataloaders['train'])
steps_per_epoch = 5
iters_per_step = iters_per_epoch // steps_per_epoch
for epoch in range(config.N_EPOCHS):
    # for phase in ['train', 'val']:
    for phase in ['train']: #Let's skip the validation in adversarial training
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        tmp_loss = []
        tmp_acc = []
        for i, (images, labels) in enumerate(dataloaders[phase]):
            images, labels = images.to(device), labels.to(device)

            # adversarial training attack - not required for testing
            if phase == 'train' and training_type == 'adversarial':
                images = attack(images, labels).to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                # Run the forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                # Store for plotting
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                loss_list[phase].append(loss.item())
                acc_list[phase].append(correct / total)
                tmp_loss.append(loss.item())
                tmp_acc.append(correct / total)

                if phase == 'train':
                    # Backprop and perform Adam optimization
                    loss.backward()
                    optimizer.step()

                    if (i + 1) % iters_per_step == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                                .format(epoch + 1, config.N_EPOCHS, int((i + 1) / iters_per_step),
                                steps_per_epoch, np.mean(tmp_loss), np.mean(tmp_acc) * 100))
                        tmp_loss = []
                        tmp_acc = []

    # Save the model at each epoch and plot
    torch.save(model.state_dict(), config.MODEL_STORE_PATH + model_name)

# Plot
window_train = len(loss_list['train']) // config.N_EPOCHS
# p = figure(x_axis_label='Epochs', y_axis_label='Loss', width=850, y_range=(0.97*min(loss_list['train']), 1.03*max(loss_list['val'])), title='{}'.format(net_type))
p = figure(x_axis_label='Epochs', y_axis_label='Loss', width=850, y_range=(0.97*min(loss_list['train']), 1.03*max(loss_list['train'])), title='{}'.format(net_type))
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')

moving_mean_loss_train = np.convolve(np.array(loss_list['train']), np.ones((window_train,))/window_train, mode='valid')
p.line(np.arange(len(moving_mean_loss_train))/len(dataloaders['train']), moving_mean_loss_train, color='blue', legend='Loss (Training)')
# p.circle(np.arange(1, config.N_EPOCHS + 1), loss_means, color='orange', legend='Loss (Validation)')
# print(np.arange(1, config.N_EPOCHS + 1), loss_means)

moving_mean_acc_train = np.convolve(np.array(acc_list['train'])*100, np.ones((window_train,))/window_train, mode='valid')
p.line(np.arange(len(moving_mean_acc_train))/len(dataloaders['train']), moving_mean_acc_train, y_range_name='Accuracy', color='red', legend='Accuracy (Training)')
# p.circle(np.arange(1, config.N_EPOCHS + 1), acc_means, y_range_name='Accuracy', color='green', legend='Accuracy (Validation)')
output_file('{}/{}_{}.html'.format(config.EXPERIMENTS_STORE_PATH, dataset, net_type))
# print(np.arange(1, config.N_EPOCHS + 1), acc_means)
save(p)
