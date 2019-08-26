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
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import mode
from sklearn.metrics import accuracy_score

# from utils import plt_confusion_matrix
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

    
    autoencoder = AutoEncoder()     

    
    if args.output=='':
        dataset = FilteredMNIST(label=args.label, split=0.8, n_noise_clusters=3)

        print(dataset.train.targets.unique(), len(dataset.train), len(dataset.test))
        autoencoder.fit(dataset,
                batch_size=args.batch_size, 
                epochs=args.epochs, 
                lr=args.lr, 
                opt='Adam',         # Adam
                loss='BCE',         # BCE or MSE
                eval=True,          # Eval training process with test data
                # plt_imgs=(N_TEST_IMGS, 5),         # (N_TEST_IMGS, plt_interval)
                scatter_plt=('tsne', 10),         # ('method', plt_interval)
                output_dir=OUTPUT_DIR, 
                save_model=True)        # Also saves dataset

    else:       # Load model and perform Kmeans
        output_dir= args.output   
        if args.output =='latest':
            output_dir = max(glob.iglob('./*/'), key=os.path.getctime)

        dataset = FilteredMNIST(output_dir=output_dir)
        model_path = autoencoder.load_model(output_dir=output_dir)
        autoencoder.eval()

        # print(dataset.test.targets.unique()) 
        
        # model_name = autoencoder.model_name
        
        # # =================== CLUSTER ASSIGNMENT ===================== #
        data = dataset.test.data.type(torch.FloatTensor)  # Loaded as FloatTensor
        data = data.view(-1, 784).to(device) # Reshape and transfer

        # encoded, decoded = autoencoder(data)
        # feat = encoded.data.cpu().view(-1, encoded.data.shape[1]).numpy()
        # print(len(feat), feat.shape)

        feat = data.cpu().numpy()
        print(len(feat), feat.shape)
        
        # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=SEED)
        # feat = tsne.fit_transform(feat)
        # pca = PCA(n_components=2, random_state=SEED)
        # feat = pca.fit_transform(feat)
        
        kmeans = KMeans(n_clusters=dataset.N_NOISE_CLUSTERS+1, n_init=20, random_state=SEED)
        y_pred = kmeans.fit_predict(feat)
        print(y_pred)
        plt.scatter(feat[:, 0], feat[:, 1], c=dataset.test.targets, s=30, cmap='viridis')

        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.savefig(output_dir+'/kmeans_imgs.png', bbox_inches='tight')

        # print(centers.shape)
        # Permute the labels
        # labels = np.zeros_like(y_pred)
        # for i in range(dataset.N_NOISE_CLUSTERS+1):
        #     mask = (y_pred == i)
        #     labels[mask] = mode(dataset.test.targets[mask])[0]

        # Rename cluster_labels to true_labels
        
        y_labels = dataset.test.targets.numpy()
        y_target_labels = np.unique(y_labels)
        y_pred_labels= np.unique(y_pred)
        # np.unique(y_target)
        print(np.unique(y_pred),  np.unique(y_labels))

        # Rename cluster labels from largest to smallest
        # print(centers.sum(axis=1))
        # lut = np.argsort(centers.sum(axis=1))[::-1]
        # lut = np.zeros_like(idx)
        # y_pred_sorted = lut[y_pred]
        # print(lut)
        # print(y_pred)
        # print(y_pred_sorted)

        # Remap to target_laels
        # lut = y_target_labels
        # print(lut)
        # y_pred_new = lut[y_pred]
        # print(y_pred_new)


        print(y_labels)
        lut = y_pred_labels
        print(lut)
        y_labels_new = lut[y_labels]
        print(y_labels_new)
        print(y_labels)
        

        # Rename 
        # print(accuracy_score(y_labels, y_pred_sorted))
        
        # y_targets_new = []
        # for y_pred in y_pred_labels:
        #     for y_label in y_labels:
        #         if y_pred == y_label
        # y_target_labels = np.unique(y_targets)
        # for i in range(dataset.N_NOISE_CLUSTERS+1):
        #     mask = (y_pred[i]==i)
        #     y_pred[mask] = y_target_labels[i]
        #     print(y_targets[i], y_target_labels[i])
        # print(np.unique(y_pred))
        
        # # =================== EVAL ACC ===================== #
        # y_target = dataset.test.targets.numpy()
        

        # print(np.unique(y_pred),  np.unique(y_target))
        accuracy, reassignment = cluster_accuracy(y_pred, y_labels_new)
        print('Accuracy: \t', accuracy)
        print('Reassignment: \t', reassignment)

        # Rename model with accuracy
        # os.rename()
        # view_data = autoencoder.data.cpu().view(-1, encoded.data.shape[1]).numpy()
        # plt_scatter(view_data, epoch, method, output_dir, pltshow=False)
        # plt_confusion_matrix(y_pred, y_target, output_dir)



        mat = confusion_matrix(y_labels_new, y_pred)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=dataset.test.targets.unique().tolist(),
                    yticklabels=dataset.test.targets.unique().tolist())
        plt.xlabel('true label')
        plt.ylabel('predicted label');

        plt.savefig(output_dir+'/confusion_matrix.png', bbox_inches='tight')

