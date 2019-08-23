#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, August 22nd 2019, 11:37:55 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Aug 23 2019
# -----
# Copyright (c) 2019 Victoria University of Wellington ECS
###

import sys
sys.path.append('..')
import os
import argparse
import glob
from datetime import datetime

import torch
from torchvision import transforms
from torchvision.datasets import MNIST

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from utils import plt_confusion_matrix
from utils import cluster_accuracy  
from ae.ae import AutoEncoder

CURRENT_FNAME = os.path.basename(__file__).split('.')[0]
timestamp = datetime.now().strftime('%Y.%d.%m-%H:%M:%S')
# Create output folder corresponding to current filename
OUTPUT_DIR = './{}_{}_output'.format(CURRENT_FNAME, timestamp)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3       
DOWNLOAD_MNIST = True
N_TEST_IMGS = 8

SEED = 489
torch.manual_seed(SEED)    # reproducible
np.random.seed(SEED)



if __name__ == '__main__':
    # Defaults set as vars above but can be overwritten by cmd line args
    parser = argparse.ArgumentParser(description='AE MNIST Example')
    parser.add_argument('--lr', type=float, default=LR, metavar='N',
                        help='learning rate for training (default: 1e-3)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--update-interval', type=int, default=100, metavar='N',
                        help='update interval for each batch')
    parser.add_argument('--epochs', type=int, default=EPOCHS, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--model', type=str, default='', metavar='N',
                        help='path/to/model or latest')
    args = parser.parse_args()

    # =================== LOAD DATA ===================== #
    # MNIST dataset - download from torchvision.datasets.mnist
    # https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
    train_data = MNIST('../',                   # Download dir
            train=True,                         # Download training data 
            transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray (H x W x C) [0, 255]
                                                # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
            download=DOWNLOAD_MNIST
            )
    test_data = MNIST('../',                     # Download dir
            train=False,                         # Download test data
            transform=transforms.ToTensor()
            )
    autoencoder = AutoEncoder()     

    # Load model and perform Kmeans
    if args.model!='':
        if args.model=='latest':
            args.model = max(glob.iglob('./*/*.pth'), key=os.path.getctime)
            autoencoder.load_model(args.model)
            autoencoder.eval()

    
        # =================== CLUSTER ASSIGNMENT ===================== #
        data = train_data.data.type(torch.FloatTensor)  # Loaded as ByteTensor
        data = data.view(-1, 784).to(device) # Reshape and transfer

        encoded, decoded = autoencoder(data)
        kmeans = KMeans(n_clusters=10, n_init=20, random_state=SEED)
        y_pred = kmeans.fit_predict(encoded.data.cpu().numpy())
        
        
        # # =================== EVAL ACC ===================== #
        y_target = train_data.targets.numpy()
        accuracy, reassignment = cluster_accuracy(y_pred, y_target)
        print('Accuracy: \t', accuracy)
        print('Reassignment: \t', reassignment)

        # plt_confusion_matrix(y_pred, y_target, OUTPUT_DIR)

    else:
        autoencoder.fit(train_data, 
                        test_data,
                        batch_size=args.batch_size, 
                        epochs=args.epochs, 
                        lr=args.lr, 
                        opt='Adam',         # Adam
                        loss='BCE',         # BCE or MSE
                        eval=True,      # Eval training process with test data
                        # n_test_imgs=N_TEST_IMGS, 
                        # scatter_plt='tsne', 
                        output_dir=OUTPUT_DIR, 
                        save_model=True)

