#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: config.py
    Author: José Hilario
    Date created: 08.11.2018
    Date last modified: 21.05.2019
    Python Version: 3.5.6
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.distributions import Normal
import torchvision.datasets
import foolbox
from foolbox.attacks import SaliencyMapAttack, DeepFoolAttack, FGSM, PGD, CarliniWagnerL2Attack
import numpy as np

from classes.BaselineCNN import BaselineCNN
from classes.CRFCNN import CRFCNN
from classes.DeepCRFCNN import DeepCRFCNN

# Settings
ADDITIVE_NOISE = True
PLOT = True
N_EPOCHS = 150
TRAINING_BATCH_SIZE = 120
VAL_BATCH_SIZE = 2000
OPTIMIZER = {
    'wd': 0.0,
    'lr': 0.01,
    'mt': 0.9
}
MODEL_STORE_PATH = './models/'
ADV_STORE_PATH = './data/adversarial/'
EXPERIMENTS_STORE_PATH = './experiments/'
ADV_MAX_ITER = 1000
CONFIDENCE_THRESHOLD = 0.2

# Nominal range of the datasets as given by Pytorch
NOMINAL_RANGE = (0, 1)
EPS = 1e-6
STATISTICS = {
    'MNIST': {
        'MEAN': 0.1307,
        'STD': 0.3081,
        'MIN': round(((NOMINAL_RANGE[0] - 0.1307) / 0.3081)*1e6)/1e6,
        'MAX': round(((NOMINAL_RANGE[1] - 0.1307) / 0.3081)*1e6)/1e6
    },
    'FashionMNIST': {
        'MEAN': 0.2860,
        'STD': 0.3202,
        'MIN': round(((NOMINAL_RANGE[0] - 0.2860) / 0.3202)*1e6)/1e6,
        'MAX': round(((NOMINAL_RANGE[1] - 0.2860) / 0.3202)*1e6)/1e6
    },
    'EMNIST': { # not computed yet
        'MEAN': 0,
        'STD': 1.,
        'MIN': ((NOMINAL_RANGE[0] - 0) / 1.) - EPS,
        'MAX': ((NOMINAL_RANGE[1] - 0) / 1.) + EPS
    },
    'CIFAR10': {
        'MEAN': (0.4914, 0.4822, 0.4465),
        'STD': (0.2470, 0.2430, 0.2610),
        'MIN': (((NOMINAL_RANGE[0] - 0.4914) / 0.2470) - EPS, ((NOMINAL_RANGE[0] - 0.4822) / 0.2430) - EPS, ((NOMINAL_RANGE[0] - 0.4465) / 0.2610) - EPS),
        'MAX': (((NOMINAL_RANGE[1] - 0.4914) / 0.2470) + EPS, ((NOMINAL_RANGE[1] - 0.4822) / 0.2430) + EPS, ((NOMINAL_RANGE[1] - 0.4465) / 0.2610) + EPS)
    }
}

DATASETS = {
    'MNIST': {
        'name': 'MNIST',
        'path': './data/MNISTData',
        'config': {
            'classes': 10,
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((STATISTICS['MNIST']['MEAN'],), (STATISTICS['MNIST']['STD'],))])
        }
    },
    'FashionMNIST': {
        'name': 'FashionMNIST',
        'path': './data/FashionMNISTData',
        'config': {
            'classes': 10,
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((STATISTICS['FashionMNIST']['MEAN'],), (STATISTICS['FashionMNIST']['STD'],))])
        }
    },
    'EMNIST': {
        'name': 'EMNIST',
        'path': './data/EMNISTData',
        'config': {
            'classes': 47,
            'split': 'balanced',
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((STATISTICS['EMNIST']['MEAN'],), (STATISTICS['EMNIST']['STD'],))])
        }
    },
    'CIFAR10': {
        'name': 'CIFAR10',
        'path': './data/CIFAR10Data',
        'config': {
            'classes': 10,
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize(STATISTICS['CIFAR10']['MEAN'], STATISTICS['CIFAR10']['STD'])])
        }
    }
}

ADV_DATASETS = {
    'MNIST': {
        'name': 'MNIST',
        'path': './data/MNISTData',
        'config': {
            'classes': 10,
            'bounds': (STATISTICS['MNIST']['MIN'], STATISTICS['MNIST']['MAX']),
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((STATISTICS['MNIST']['MEAN'],), (STATISTICS['MNIST']['STD'],))]),
            'preprocessing': (STATISTICS['MNIST']['MEAN'], STATISTICS['MNIST']['STD'])
        },
        'statistics': {
            'mean': STATISTICS['MNIST']['MEAN'],
            'std': STATISTICS['MNIST']['STD']
        }
    },
    'FashionMNIST': {
        'name': 'FashionMNIST',
        'path': './data/FashionMNISTData',
        'config': {
            'classes': 10,
            'bounds': (STATISTICS['FashionMNIST']['MIN'], STATISTICS['FashionMNIST']['MAX']),
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((STATISTICS['FashionMNIST']['MEAN'],), (STATISTICS['FashionMNIST']['STD'],))]),
            'preprocessing': (STATISTICS['FashionMNIST']['MEAN'], STATISTICS['FashionMNIST']['STD'])
        },
        'statistics': {
            'mean': STATISTICS['FashionMNIST']['MEAN'],
            'std': STATISTICS['FashionMNIST']['STD']
        }
    },
    'CIFAR10': {
        'name': 'CIFAR10',
        'path': './data/CIFAR10Data',
        'config': {
            'classes': 10,
            'bounds': (STATISTICS['CIFAR10']['MIN'], STATISTICS['CIFAR10']['MAX']),
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((STATISTICS['CIFAR10']['MEAN'],), (STATISTICS['CIFAR10']['STD'],))]),
            'preprocessing': (STATISTICS['CIFAR10']['MEAN'], STATISTICS['CIFAR10']['STD'])
        },
        'statistics': {
            'mean': STATISTICS['CIFAR10']['MEAN'],
            'std': STATISTICS['CIFAR10']['STD']
        }
    }
}

ANALYSIS_DATASETS = {
    'MNIST': {
        'name': 'MNIST',
        'path': './data/MNISTData',
        'config': {
            'classes': 10,
            'transform': transforms.Compose([transforms.ToTensor()])
        }
    },
    'FashionMNIST': {
        'name': 'FashionMNIST',
        'path': './data/FashionMNISTData',
        'config': {
            'classes': 10,
            'transform': transforms.Compose([transforms.ToTensor()])
        }
    },
    'EMNIST': {
        'name': 'EMNIST',
        'path': './data/EMNISTData',
        'config': {
            'classes': 47,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'split': 'balanced'
        }
    },
    'CIFAR10': {
        'name': 'CIFAR10',
        'path': './data/CIFAR10Data',
        'config': {
            'classes': 10,
            'transform': transforms.Compose([transforms.ToTensor()])
        }
    }
}

VISUALIZATION_DATASETS = {
    'MNIST': {
        'name': 'MNIST',
        'path': './data/MNISTData',
        'config': {
            'classes': 10,
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((STATISTICS['MNIST']['MEAN'],), (STATISTICS['MNIST']['STD'],))]),
            'bounds': (STATISTICS['MNIST']['MIN'], STATISTICS['MNIST']['MAX']),
            'preprocessing': (STATISTICS['MNIST']['MEAN'], STATISTICS['MNIST']['STD'])
        },
        'statistics': {
            'mean': STATISTICS['MNIST']['MEAN'],
            'std': STATISTICS['MNIST']['STD']
        }
    },
    'FashionMNIST': {
        'name': 'FashionMNIST',
        'path': './data/FashionMNISTData',
        'config': {
            'classes': 10,
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((STATISTICS['FashionMNIST']['MEAN'],), (STATISTICS['FashionMNIST']['STD'],))]),
            'bounds': (STATISTICS['FashionMNIST']['MIN'], STATISTICS['FashionMNIST']['MAX']),
            'preprocessing': (STATISTICS['FashionMNIST']['MEAN'], STATISTICS['FashionMNIST']['STD'])
        },
        'statistics': {
            'mean': STATISTICS['FashionMNIST']['MEAN'],
            'std': STATISTICS['FashionMNIST']['STD']
        }
    },
    'CIFAR10': {
        'name': 'CIFAR10',
        'path': './data/CIFAR10Data',
        'config': {
            'classes': 10,
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((STATISTICS['CIFAR10']['MEAN'],), (STATISTICS['CIFAR10']['STD'],))]),
            'bounds': (STATISTICS['CIFAR10']['MIN'], STATISTICS['CIFAR10']['MAX']),
            'preprocessing': (STATISTICS['CIFAR10']['MEAN'], STATISTICS['CIFAR10']['STD'])
        },
        'statistics': {
            'mean': STATISTICS['CIFAR10']['MEAN'],
            'std': STATISTICS['CIFAR10']['STD']
        }
    }
}
FEATURES_BATCH_SIZE = 15

ADV_TRAINING_SETUP = {
  # The perturbations reported in Madry's paper (and others) are with respect to normalized input between [0,1]
  # therefore, it is necessary to scale it to the range of our inputs.
    'MNIST': {
        'epsilon': 0.3 / STATISTICS['MNIST']['STD'],
        'k': 10,
        'step_size': 0.01,
        'random_start': True,
        'range': (STATISTICS['MNIST']['MIN'], STATISTICS['MNIST']['MAX'])
    },
    'FashionMNIST': {
        'epsilon': 0.3 / STATISTICS['FashionMNIST']['STD'],
        'k': 10,
        'step_size': 0.01,
        'random_start': True,
        'range': (STATISTICS['FashionMNIST']['MIN'], STATISTICS['FashionMNIST']['MAX'])  
    }
}

NETWORK_PARAMETERS = {
    'BaselineCNN': {
        'train': {
            'MNIST': { 'c': 1, 'n': 10, 'std': 0.1},
            'FashionMNIST': { 'c': 1, 'n': 10, 'std': 0.1},
            'CIFAR10': { 'c': 3, 'n': 10, 'std': 0.1}
        },
        'test': {
            'MNIST': { 'c': 1, 'n': 10, 'std': 0.},
            'FashionMNIST': { 'c': 1, 'n': 10, 'std': 0.},
            'CIFAR10': { 'c': 3, 'n': 10, 'std': 0.}
        }
    },
    'CRFCNN': {
        'train': {
            'MNIST': { 'c': 1, 'n': 10, 'std': 0.1},
            'FashionMNIST': { 'c': 1, 'n': 10, 'std': 0.1}
        },
        'test': {
            'MNIST': { 'c': 1, 'n': 10, 'std': 0.},
            'FashionMNIST': { 'c': 1, 'n': 10, 'std': 0.}
        }
    },
    'DeepCRFCNN': {
        'train': { 'MNIST': { 'n': 10, 'std': 0.1} },
        'test': { 'MNIST': { 'n': 10, 'std': 0.} }
    }
}

# Hardcoded parameters
def attacks(name):
    return {
        'saliency_map': SaliencyMapAttack(),
        'deepfool': DeepFoolAttack(),
        'fgsm': FGSM(),
        'pgd': PGD(),
        'cw': CarliniWagnerL2Attack()
    }[name]

def network_model(name, dataset, mode):
    return {
        'BaselineCNN': BaselineCNN(**NETWORK_PARAMETERS[name][mode][dataset]),
        'CRFCNN': CRFCNN(**NETWORK_PARAMETERS[name][mode][dataset]),
        'DeepCRFCNN': DeepCRFCNN(**NETWORK_PARAMETERS[name][mode][dataset])
    }[name]
