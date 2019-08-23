#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, August 22nd 2019, 6:21:22 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Thu Aug 22 2019
# -----
# Copyright (c) 2019 Victoria University of Wellington ECS
###

import sys
sys.path.append("..")
import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils import scatter

CURRENT_FNAME = os.path.basename(__file__).split('.')[0]

# Create output folder corresponding to current filename
OUTPUT_DIR = './' + CURRENT_FNAME + '_output'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
N_EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 8

# =================== LOAD DATA ===================== #
# MNIST dataset
# Download from torchvision.datasets.mnist
# https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
train_data = MNIST('../',                      # Download dir
        train=True,                         # Download training data only
        transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray (H x W x C) [0, 255]
                                            # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        download=DOWNLOAD_MNIST
        )
test_data = MNIST('../',                      # Download dir
        train=False,                         # Download training data only
        transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray (H x W x C) [0, 255]
                                            # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        download=DOWNLOAD_MNIST
        )

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# =================== MODEL ===================== #
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 10),   # compress to 10 features, can use further method to vis
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)           # z
        decoded = self.decoder(encoded)     # recon_x (x_hat)
        return encoded, decoded


# =================== TRAIN ===================== #
def train(epoch):
    autoencoder.train() # instantiate nn.Module in train mode
    epoch_loss = 0      # for printing intermediary loss
    for batch_idx, (batch_train, _) in enumerate(train_loader):
        batch_train = batch_train.to(device)                # moving batch to GPU if available
        batch_x = batch_train.view(batch_train.size(0), -1)   # batch y, shape (batch, 28*28)
        batch_y = batch_train.view(batch_train.size(0), -1)   # batch y, shape (batch, 28*28)
        
        # =================== forward ===================== #
        encoded, decoded = autoencoder(batch_x)
        loss = loss_func(decoded, batch_y)      
        MSE_loss = nn.MSELoss()(decoded, batch_y)   # mean square error
        # =================== backward ==================== #
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        epoch_loss += loss.data
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss:{:.6f} \t MSE Loss:{:.6f} '.format(
                epoch, batch_idx * len(batch_train), len(train_loader.dataset),
                       100.0 * batch_idx / len(train_loader),
                       loss.data / len(batch_train),
                       MSE_loss.data / len(batch_train)
            ))
            
    epoch_loss /= len(train_loader.dataset)
    print('\n====> Epoch: {} Average loss: {:.4f}'.format(epoch, epoch_loss))
    return epoch_loss

# =================== TEST ===================== #
def test(epoch):
    autoencoder.eval()        # instantiate nn.Module in test mode
    test_loss = 0   
    feat_total = []
    target_total = []

    with torch.no_grad():      # turn autograd off for memory efficiency
        for batch_idx, (batch_test, batch_test_label) in enumerate(test_loader):
            batch_test = batch_test.to(device)
            batch_test = batch_test.view(batch_test.size(0), -1)  
            encoded, decoded = autoencoder(batch_test)

            # Accumulate test loss, feat_total and target_total
            test_loss += loss_func(decoded, batch_test.view(batch_test.size(0), -1)).data
            feat_total.append(encoded.data.cpu().view(-1, encoded.data.shape[1]))
            target_total.append(batch_test_label)

            # =================== PLOT COMPARISON ===================== #
            if batch_idx == 0:
                # Reshape into (N_TEST_IMG, 1, 28, 28)
                batch_test = batch_test.view(-1, 1, 28, 28)    
                decoded = decoded.view(-1, 1, 28, 28)

                comparison = torch.cat([batch_test[:N_TEST_IMG], decoded.view(-1, 1, 28, 28)[:N_TEST_IMG]])
                save_image(comparison.data.cpu(),
                        OUTPUT_DIR+'/x_recon_' + str(epoch) + '.png', nrow=N_TEST_IMG)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}\n'.format(test_loss))

        # =================== PLOT SCATTER ===================== #
        feat_total = torch.cat(feat_total, dim=0)
        target_total = torch.cat(target_total, dim=0)
        # scatter(feat_total.numpy(), target_total.numpy(), epoch, 'pca', OUTPUT_DIR)
        # scatter(feat_total.numpy(), target_total.numpy(), epoch, 'tsne', OUTPUT_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AE MNIST Example')
    parser.add_argument('--lr', type=float, default=LR, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--update-interval', type=int, default=100, metavar='N',
                        help='update interval for each batch')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--weights', type=str, default="", metavar='N',
                    help='path to weights')
    args = parser.parse_args()

    autoencoder = AutoEncoder().to(device)      # moving ae to GPU if available

    # TODO: Investigate weight decay
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)

    # TODO: Investigate BCE or MSE loss
    loss_func = nn.BCELoss()
    # loss_func = nn.MSELoss()
    for epoch in range(N_EPOCHS+1):
        train(epoch)
        test(epoch)
    
    # Saving model weights
    torch.save(autoencoder.state_dict(), './' + CURRENT_FNAME + '_weights.pth')

