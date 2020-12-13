#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    File name: Loader.py
    Author: José Hilario
    Date created: 08.11.2018
    Date last modified: 21.05.2019
    Python Version: 3.5.6
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.distributions import Normal
import torchvision.datasets
import inspect
import glob
from PIL import Image
import numpy as np

class Loader(object):
    def __init__(self):
        pass

    def get_keyword_arguments(self, fn):
        """
        Parameters
        ----------
        fn: Function
            function from which to get keyword arguments

        Returns
        -------
        tuple(list,list,list,list)
            arguments of the function
        """
        return inspect.getargspec(fn)

    def get_function_parameters(self, fn, *args, **kwargs):
        """
        Parameters
        ----------
        fn: Function
            function from which to get keyword arguments
        args:
        kwargs:

        Returns
        -------
        tuple(list, list)
            filtered arguments to pass to the function fn
        """
        fn_args, _, _, _ = inspect.getargspec(fn)
        res_kwargs = kwargs
        unwanted = set(kwargs) - set(fn_args)
        for unwanted_key in unwanted: del res_kwargs[unwanted_key]
        return args, res_kwargs

    def load_dataset(self, dataset, path, *args, **kwargs):
        """
        Parameters
        ----------
        dataset: string
            name of dataset to be loaded
        path: string
            path of dataset to be loaded
        args:
        kwargs:

        Returns
        -------
        Dataset <dataset>
            dataset object requested
        """
        args, kwargs = self.get_function_parameters(getattr(torchvision.datasets, dataset), *args, **kwargs)
        return getattr(torchvision.datasets, dataset)(root=path, *args, **kwargs)

    def load_train_dataset(self, dataset, path, *args, **kwargs):
        args, kwargs = self.get_function_parameters(getattr(torchvision.datasets, dataset), *args, **kwargs)
        return getattr(torchvision.datasets, dataset)(root=path, train=True, download=True, *args, **kwargs)

    def load_test_dataset(self, dataset, path, *args, **kwargs):
        args, kwargs = self.get_function_parameters(getattr(torchvision.datasets, dataset), *args, **kwargs)
        return getattr(torchvision.datasets, dataset)(root=path, train=False, download=True, *args, **kwargs)

    def load_adversarial_dataset(self, dataset, path, preprocessing):
        mean, std = preprocessing
        dataset = []

        for filename in sorted(glob.glob(path + '/*.png'), key=lambda e: (len(e), e)):
            img = Image.open(filename)
            arr = torch.from_numpy(np.array(img.getdata())).float() / 255.
            arr = arr.reshape((1, img.size[0], img.size[1]))
            dataset.append(F.normalize(arr, (mean,), (std,)))

        f = open(path + '/labels.csv')
        labels, distances = [], []
        for line in f.readlines():
            line = line.split(', ')
            labels.append(int(line[1]))
            distances.append(float(line[2].strip('\n')))

        return torch.stack(dataset), torch.tensor(labels), distances
