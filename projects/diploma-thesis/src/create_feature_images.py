#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: create_feature_images.py
    Author: José Hilario
    Date created: 20.04.2019
    Date last modified: 21.05.2019
    Python Version: 3.5.6
'''

# Libraries
import sys
import os
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torchvision.transforms import functional as F
import numpy as np
import foolbox
from foolbox.adversarial import Adversarial
from foolbox.criteria import Misclassification, ConfidentMisclassification
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
from PIL import Image, ImageEnhance

# Modules
from classes.Loader import Loader
import config

# Script arguments
model_names = sys.argv[1].split(',')
net_types = sys.argv[2].split(',')
dataset = sys.argv[3]
n_batches = int(sys.argv[4])
experiment_name = sys.argv[5]
attack_name = sys.argv[6]

# Data loader
loader = Loader()
test_dataset = loader.load_test_dataset(dataset=config.VISUALIZATION_DATASETS[dataset]['name'], path=config.VISUALIZATION_DATASETS[dataset]['path'], **config.VISUALIZATION_DATASETS[dataset]['config'])
test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.FEATURES_BATCH_SIZE, shuffle=False)

# ConvNet
device = torch.device('cpu')
criterion = nn.NLLLoss()
models, fmodels = {}, {}
for model_name, net_type in zip(model_names, net_types):
    model = config.network_model(net_type, dataset, 'test').to(device)
    model.load_state_dict(torch.load(config.MODEL_STORE_PATH + model_name, map_location=device))
    model.eval()
    models[model_name] = model
    fmodels[model_name] = foolbox.models.PyTorchModel(
        model, bounds=config.VISUALIZATION_DATASETS[dataset]['config']['bounds'], num_classes=config.VISUALIZATION_DATASETS[dataset]['config']['classes'])

def lazy_create_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def normalize_image(image, bounds=None, k=1):
    if bounds is None:
        out = image - image.min()
        return k * out / out.max()
    else:
        out = image - bounds[0]
        return k * out / (bounds[1] - bounds[0])

def filter_image(image):
    out = image.clone()
    # out[0, out[0] < out[0].median()] = 0
    # out[1, out[1] < out[1].median()] = 0
    # out[2, out[2] < out[2].median()] = 0
    # print(out.min(), out.max())
    # print(out.min(), out.max(), out.mean(), out.median())
    out[0, out[0] < out[0].max()*0.1] = 0
    out[1, out[1] < out[1].max()*0.1] = 0
    out[2, out[2] < out[2].max()*0.1] = 0
    # print(out.min(), out.max(), out.mean(), out.median())
    return out

def get_with_white_background(image):
    out = image.clone()
    out[:, (out[0] == 0) & (out[1] == 0) & (out[2] == 0)] = 1
    return out

def save_image_batch(images, path, bounds=None):
    for i, image in enumerate(images):
        image_values = normalize_image(image, bounds, 255).numpy()
        result = Image.fromarray(image_values.astype('uint8')[0])
        result.save('{}/orig_{}.png'.format(path, i), 'PNG', compress_level=0)

def save_gradient_batch(images, path):
    lazy_create_path(path)
    N, C, H, W = images.size()
    for i, image in enumerate(images):
        out = torch.zeros(torch.Size((3, H, W)))
        if C == 1:
            # RGB
            out[0, image[0] < 0] = -image[0, image[0] < 0]
            # out[1, image[0] == 0] = 0 #image[0, image[0] == 0]
            out[2, image[0] > 0] = image[0, image[0] > 0]
        else:
            out = image
        thr_image = torch.clamp(2*filter_image(normalize_image(out, None, 255)), 0, 255)
        image_values = thr_image.numpy().transpose(1,2,0).astype('uint8')
        result = Image.fromarray(image_values)
        result.save('{}/{}.png'.format(path, i), 'PNG', compress_level=0)

# Create path if doesn't exist
experiment_dir = '/'.join([str(s).strip('/') for s in [config.EXPERIMENTS_STORE_PATH, experiment_name]])
lazy_create_path(experiment_dir)
for i, (images, labels) in enumerate(test_dataloader):
    images, labels = images.to(device), labels.to(device)
    if i >= n_batches: break
    # denormalized = (images * config.VISUALIZATION_DATASETS[dataset]['statistics']['std']) + config.VISUALIZATION_DATASETS[dataset]['statistics']['mean']
    denormalized = images
    for model_name, model in models.items():
        outputs = model(denormalized)
        # print(outputs.data)
        # _, predicted = torch.max(outputs.data, 1)
        # print('{} Errors'.format((predicted != labels).sum().item()))
        denormalized.requires_grad_()
        with torch.enable_grad():
            output = model(denormalized)
            _, predicted = torch.max(outputs.data, 1)
            print(torch.exp(output[predicted == labels]).topk(5))
            loss = criterion(output, labels)
        grad = torch.autograd.grad(loss, [denormalized])[0]
        # print(grad.min(), grad.max())
        path = '/'.join([str(s).strip('/') for s in [experiment_dir, model_name, i]])
        save_gradient_batch(grad.detach(), path)
        save_image_batch(denormalized.detach(), path, config.VISUALIZATION_DATASETS[dataset]['config']['bounds'])

for i, (images, labels) in enumerate(test_dataloader):
    images, labels = images.to(device), labels.to(device)
    if i >= n_batches: break
    for model_name, fmodel in fmodels.items():
        model = models[model_name]
        batch_adv, batch_labels = None, None
        for j in range(len(images)):
            image, label = images[j].numpy(), labels[j].numpy()
            # _, predicted = torch.max(outputs.data, 1)
            # print(label, torch.argmax(model(images[j:(j+1)]).data).item())
            adversarial = Adversarial(fmodel, Misclassification(), image, label, distance=foolbox.distances.Linf)
            attack = config.attacks(attack_name)
            try:
                attack(adversarial)
                # print(model_name, j, adversarial.distance.value, adversarial.original_class, adversarial.adversarial_class)
                if adversarial.image is not None and adversarial.distance.value > 0:
                    print('Adversarial Image for {} was found with distance {} from its original'.format(model_name, adversarial.distance.value))
                    adv_images = torch.from_numpy(np.array([adversarial.image]))
                    adv_labels = torch.from_numpy(np.array([label]))
                    if batch_adv is None:
                        batch_adv = adv_images
                        batch_labels = adv_labels
                    else:
                        batch_adv = torch.cat((batch_adv, adv_images))
                        batch_labels = torch.cat((batch_labels, adv_labels))
                else:
                    print('No exception, but no adversarial. {}'.format(adversarial.distance.value))
            except:
                print('Exception')
                pass

        path = '/'.join([str(s).strip('/') for s in [experiment_dir, model_name, 'adv_{}'.format(i)]])
        # denormalized = (batch_adv.detach() * config.VISUALIZATION_DATASETS[dataset]['statistics']['std']) + config.VISUALIZATION_DATASETS[dataset]['statistics']['mean']
        denormalized = batch_adv.detach()
        if batch_adv is not None:
            denormalized.requires_grad_()
            with torch.enable_grad():
                output = model(denormalized)
                print(torch.exp(output).topk(5))
                loss = criterion(output, batch_labels)
            grad = torch.autograd.grad(loss, [denormalized])[0]
            save_gradient_batch(grad.detach(), path)
            save_image_batch(denormalized.detach(), path, config.VISUALIZATION_DATASETS[dataset]['config']['bounds'])
