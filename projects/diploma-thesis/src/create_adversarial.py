#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: create_adversarial.py
    Author: José Hilario
    Date created: 08.11.2018
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
from PIL import Image

# Modules
from classes.Loader import Loader
import config

# Script arguments
model_id = sys.argv[1]
dataset = sys.argv[2]
network_name = sys.argv[3]
adv_name = sys.argv[4]
attack_name = sys.argv[5]

# Data loader
loader = Loader()
test_dataset = loader.load_test_dataset(dataset=config.ADV_DATASETS[dataset]['name'], path=config.ADV_DATASETS[dataset]['path'], **config.ADV_DATASETS[dataset]['config'])
test_loader = DataLoader(dataset=test_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=False)

# ConvNet
device = torch.device('cpu')
model = config.network_model(network_name, dataset, 'test').to(device)
model.load_state_dict(torch.load(config.MODEL_STORE_PATH + model_id, map_location=device))
model.eval()
fmodel = foolbox.models.PyTorchModel(
    model, bounds=config.ADV_DATASETS[dataset]['config']['bounds'], num_classes=config.ADV_DATASETS[dataset]['config']['classes'])

def normalize_image(image, bounds, k=1):
    out = image - bounds[0]
    return k * out / (bounds[1] - bounds[0])

def save_image_batch(images, originals, path, order, bounds):
    for i, image in enumerate(images):
        image_values = normalize_image(image, bounds, 255)
        # This is in order to round in the direction of the gradient
        image_values[originals[i] > image] = np.floor(image_values[originals[i] > image])
        image_values[originals[i] < image] = np.ceil(image_values[originals[i] < image])
        # END
        result = Image.fromarray(image_values.numpy().astype('uint8')[0])
        order_i = np.floor(np.log10(i if i > 0 else 1)) + 1
        name = '{}.png'.format(int(order - order_i)*'0' + str(i))
        result.save('{}/{}'.format(path, name), 'PNG', compress_level=0)

# Attack on source images
out = ''
count = 0
no_adv_found = 0
batch_adv, batch_originals = None, None
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    for i in range(len(images)):
        # print('Attempt #{}'.format(i))
        image, label = images[i].numpy(), labels[i].numpy()
        adversarial = Adversarial(fmodel, Misclassification(), image, label, distance=foolbox.distances.Linf)
        attack = config.attacks(attack_name)
        try:
            attack(adversarial)
            if adversarial.image is not None and adversarial.distance.value > 0:
                count += 1
                # Add only when an adversarial has been found for an otherwise correctly classified image
                print('Adversarial Image #{} was found with distance {} from its original'.format(count, adversarial.distance.value))
                out += ', '.join([str(count), str(label), str(adversarial.distance.value)]) + '\r\n'
                adv_images = torch.from_numpy(np.array([adversarial.image]))
                originals = torch.from_numpy(np.array([image]))
                if batch_adv is None:
                    batch_adv = adv_images
                    batch_originals = originals
                else:
                    batch_adv = torch.cat((batch_adv, adv_images))
                    batch_originals = torch.cat((batch_originals, originals))
                if count >= config.ADV_MAX_ITER: break
            else:
                if adversarial.distance.value == np.inf:
                    no_adv_found += 1
        except: pass

print('{} failed attempts to find an adversarial'.format(no_adv_found))
# Save original labels of images for which adversarials were found
path = '/'.join([str(s).strip('/') for s in [config.ADV_STORE_PATH, '{}_{}'.format(adv_name, attack_name)]])
if not os.path.isdir(path):
    os.makedirs(path)

f = open(path + '/labels.csv', 'w')
f.write(out)
f.close()

order = np.floor(np.log10(len(adv_images))) + 1
save_image_batch(batch_adv, batch_originals, path, order, config.ADV_DATASETS[dataset]['config']['bounds'])
