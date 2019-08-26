#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Monday, August 26th 2019, 12:13:26 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Aug 26 2019
# -----
# Copyright (c) 2019 Victoria University of Wellington ECS
###

import numpy as np
import random

import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from sklearn.model_selection import train_test_split

class FilteredMNIST(Dataset):
    def __init__(self, label=0, split=0.8, n_noise_clusters=2,  output_dir=""):
        super(Dataset, self).__init__()

        # If training
        if output_dir=="":
            self.LABEL = label
            self.N_NOISE_CLUSTERS = n_noise_clusters
            self.SPLIT= split
            self.train, self.test = self._load_filtered_mnist()

    
    def _load_filtered_mnist(self):
                # =================== LOAD DATA ===================== #
        # MNIST dataset - download from torchvision.datasets.mnist
        # https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
        mnist_train = MNIST('../',                   # Download dir
                train=True,                         # Download training data 
                transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray (H x W x C) [0, 255]
                                                    # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
                download=True
                )

        mnist_test = MNIST('../',                     # Download dir
                train=False,                         # Download test data
                transform=transforms.ToTensor()
                )
        
        mnist_dataset = self._concat_dataset(mnist_train, mnist_test)
        
        noise_data, noise_targets = self._gen_noise_data(mnist_dataset)
        filtered_mnist = self._filter_label(mnist_dataset, self.LABEL)
        # print(train_filtered.targets, len(train_filtered.data ),len(train_filtered.targets))
        filtered_mnist = self._add_noise(filtered_mnist, noise_data, noise_targets)
        
        return self._split_train_test(filtered_mnist, mnist_test, self.SPLIT)

    
    def _concat_dataset(self, train, test):
        # Append test to train 
        train.data = torch.cat((train.data, test.data), dim=0)
        train.targets = torch.cat((train.targets, test.targets), dim=0)
        return train
        
    def _filter_label(self, dataset, label):
        dataset.data = dataset.data[dataset.targets == label]
        dataset.targets = dataset.targets[dataset.targets == label]
        return dataset

    def _add_noise(self, filtered,  noise_data, noise_targets):
        filtered.data = torch.cat((filtered.data, noise_data), dim=0)
        filtered.targets = torch.cat((filtered.targets, noise_targets), dim=0)
        return filtered

    def _gen_noise_data(self, dataset):
        # Randomly sampling 3 random cluster labels and removing from list
        labels = dataset.targets[dataset.targets!=self.LABEL].unique().tolist()
        rand_labels = [labels.pop(random.randrange(len(labels))) for _ in range(self.N_NOISE_CLUSTERS)]
        print(labels, rand_labels)
        # print(noise_targets.type(), targets.type())
        noise_data = torch.ByteTensor()      # Init noise and targets 
        noise_targets = torch.LongTensor()
        for i, label in enumerate(rand_labels):
            # # Get noise data and labels
            data = dataset.data[dataset.targets==label]
            targets = dataset.targets[dataset.targets==label]
            # print(noise_targets, len(noise_targets),len(noise_data))

            # Gen clusters in increasing size
            size = int(len(data)/(10-(i+1)))   
            # Get random subset of noise_data
            data = data[np.random.randint(data.numpy().shape[0], size=size)]
            targets = targets[:size] 
        
            # Append to noise 
            noise_data = torch.cat((noise_data, data), dim=0)
            noise_targets = torch.cat((noise_targets, targets), dim=0)

        print(noise_targets, len(noise_targets),len(noise_data))
        return noise_data, noise_targets
    
    def _split_train_test(self, full, test, split):
        train_size = int(split * len(full))
        test_size = len(full) - train_size
        # print(full.targets.unique(), len(full), train_size, test_size)

        train_idx, test_idx = train_test_split(    # Stratified split
            np.arange(len(full.targets)), test_size=test_size, random_state=489, shuffle=True, stratify=full.targets)
    
        test.data = full.data[test_idx]  
        test.targets = full.targets[test_idx]
        full.data = full.data[train_idx]
        full.targets = full.targets[train_idx]
    
        # print(full.targets.unique(), test.targets.unique(), len(full), len(test))
        return full, test

    def save_dataset(self, path):
        torch.save(self, path)