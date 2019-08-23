#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, August 22nd 2019, 5:13:17 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Thu Aug 22 2019
# -----
# Copyright (c) 2019 Victoria University of Wellington ECS
###

import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# torch.manual_seed(1)    # reproducible

# Create output folder corresponding to current filename
OUTPUT_DIR = './' + os.path.basename(__file__).split('.')[0] + '_output'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Hyper Parameters
N_EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

# MNIST dataset
# Download from torchvision.datasets.mnist
# https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
train_data = MNIST('../',                      # Download dir
        train=True,                         # Download training data only
        transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray (H x W x C) [0, 255]
                                            # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        download=DOWNLOAD_MNIST
        )

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3),   # compress to 3 features which can be visualiaed in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)           # z
        decoded = self.decoder(encoded)     # recon_x (x_hat)
        return encoded, decoded


# =================== PLOT IMAGE =====================

# plot one example
# print(train_data.data.size())     # (60000, 28, 28)
# print(train_data.targets.size())   # (60000)
# plt.imshow(train_data.data[2].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[2])
# plt.show()

# initialise figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot
# original data (first row) for viewing
view_data = train_data.data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.0
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.cpu().data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())


# =================== TRAIN =====================
autoencoder = AutoEncoder().to(device)      # moving ae to GPU if available

# TODO: Investigate weight decay
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)

# TODO: Investigate BCE or MSE loss
loss_func = nn.BCELoss()
# loss_func = nn.MSELoss()


for epoch in range(N_EPOCHS+1):
    for batch_idx, (batch, batch_label) in enumerate(train_loader):
        batch = batch.to(device)                # moving batch to GPU if available
        batch_x = batch.view(batch.size(0), -1)   # batch x, shape (batch, 28*28)
        batch_y = batch.view(batch.size(0), -1)   # batch y, shape (batch, 28*28)
        
        # =================== forward =====================
        encoded, decoded = autoencoder(batch_x)
        loss = loss_func(decoded, batch_y)      
        MSE_loss = nn.MSELoss()(decoded, batch_y)   # mean square error
        # =================== backward ====================
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if batch_idx % 100 == 0:
            print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
                .format(epoch, N_EPOCHS, loss.data, MSE_loss.data))

    # =================== PLOT =====================
    # plotting decoded image (second row) 
    if (epoch % 5 == 0):    
        _, decoded_data = autoencoder(view_data.to(device))
        for i in range(N_TEST_IMG):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decoded_data.cpu().data.numpy()[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(()); a[1][i].set_yticks(())
        plt.draw(); 
        # plt.pause(0.05)
        plt.savefig(OUTPUT_DIR+'/epoch_{}.png'.format(epoch), bbox_inches='tight')

plt.ioff()
# plt.show()


# Saving model weights
torch.save(autoencoder.state_dict(), './' + os.path.basename(__file__).split('.')[0] + '_weights.pth')


# =================== PLOT 3D =====================
# visualise in 3D plot
view_data = train_data.data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data.to(device))
encoded_data = encoded_data.cpu()   # For plotting if on ae on GPU
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.targets[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.savefig(OUTPUT_DIR+'/3D_cluster_result.png', bbox_inches='tight')
# plt.show()