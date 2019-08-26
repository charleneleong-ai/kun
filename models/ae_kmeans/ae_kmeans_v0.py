#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, August 22nd 2019, 11:37:55 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Aug 26 2019
# -----
# Copyright (c) 2019 Victoria University of Wellington ECS
###

import sys
sys.path.append('..')
import os
import argparse
import glob
from datetime import datetime
import random

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import SubsetRandomSampler, DataLoader

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from utils import plt_confusion_matrix
from utils import cluster_accuracy  
from ae.ae import AutoEncoder
from datasets import FilteredMNIST

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
random.seed(SEED)


if __name__ == '__main__':
    # Defaults set as vars above but can be overwritten by cmd line args
    parser = argparse.ArgumentParser(description='AE MNIST Example')
    parser.add_argument('--lr', type=float, default=LR, metavar='N',
                        help='learning rate for training (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--update_interval', type=int, default=100, metavar='N',
                        help='update interval for each batch')
    parser.add_argument('--epochs', type=int, default=EPOCHS, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--output', type=str, default='', metavar='N',
                        help='path/to/output/dir or latest')
    parser.add_argument('--label', type=int, default=1, metavar='N',
                        help='class to filter')
    args = parser.parse_args()

    

            
    # noise_data, noise_targets = noise_dataset(train_data, args.label, 2)
    # train_data_filtered = filter_label(train_data, args.label)
    # # print(train_data_filtered.targets, len(train_data_filtered.data ),len(train_data_filtered.targets))

    # # Add noise
    # train_data_filtered.data = torch.cat((train_data_filtered.data, noise_data), 0)
    # train_data_filtered.targets = torch.cat((train_data_filtered.targets, noise_targets), 0)


    autoencoder = AutoEncoder()     

    # Load model and perform Kmeans
    if args.output=='':
        dataset = FilteredMNIST(label=args.label, split=0.8, n_noise_clusters=3)
        print(dataset.train.targets.unique(), len(dataset.train), len(dataset.test))
        # autoencoder.fit(train_data_filtered, 
        #         train_data_filtered,
        #         batch_size=args.batch_size, 
        #         epochs=args.epochs, 
        #         lr=args.lr, 
        #         opt='Adam',         # Adam
        #         loss='BCE',         # BCE or MSE
        #         eval=True,          # Eval training process with test data
        #         plt_imgs=(N_TEST_IMGS, 5),         # (N_TEST_IMGS, plt_interval)
        #         scatter_plt=('pca', 10),         # ('method', plt_interval)
        #         output_dir=OUTPUT_DIR, 
        #         save_model=True)        # Also saves dataset
    else:
        output_dir= args.output 
        
        if args.output =='latest':
            output_dir = max(glob.iglob('./*/'), key=os.path.getctime)
            # model_path = output_dir.format()

        
        
        # train_data_filtered = FilteredMNIST()
        
        # print(train_data_filtered.targets)
        # autoencoder.load_model(output_dir)
        # autoencoder.eval()

        # model_name = autoencoder.model_name
        
        # # =================== CLUSTER ASSIGNMENT ===================== #
        # data = train_data_filtered.data.type(torch.FloatTensor)  # Loaded as ByteTensor
        # data = data.view(-1, 784).to(device) # Reshape and transfer

        # encoded, decoded = autoencoder(data)
        # n_clusters=3
        # kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=SEED)
        # y_pred = kmeans.fit_predict(encoded.data.cpu().numpy())

        # # idx = np.argsort(y_pred.cluster_centers_.sum(axis=1))
        # # lut = np.zeros_like(idx)
        # # lut[idx] = np.arange(n_clusters)
        
        # # # =================== EVAL ACC ===================== #
        # y_target = train_data.targets.numpy()
        # accuracy, reassignment = cluster_accuracy(y_pred, y_target)
        # print('Accuracy: \t', accuracy)
        # print('Reassignment: \t', reassignment)

        # # Rename model with accuracy
        # # os.rename()
        # # view_data = autoencoder.data.cpu().view(-1, encoded.data.shape[1]).numpy()
        # # plt_scatter(view_data, epoch, method, output_dir, pltshow=False)
        # plt_confusion_matrix(y_pred, y_target, model_path.split('/')[1])


