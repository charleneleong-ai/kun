#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, August 22nd 2019, 11:50:30 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Tue Aug 27 2019
# -----
# Copyright (c) 2019 Victoria University of Wellington ECS
###

import os
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision.utils import save_image

from utils import plt_scatter


class AutoEncoder(nn.Module):
    def __init__(self, ):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 500),
            nn.ReLU(inplace=True),  # modify input directly     
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 10),   # compress to 10 features, can use further method to vis
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 28*28),
            nn.Sigmoid(),           # compress to a range (0, 1)
        )

        
        # Initialise the weights and biases in the layers
        self.apply(self._init_weights)

        # Shifting to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, batch):                   # batch=x
        encoded = self.encoder(batch)           # z
        decoded = self.decoder(encoded)         # recon_x (x_hat)
        return encoded, decoded
    
    def fit(self, dataset, batch_size, epochs, lr, opt='Adam', loss='BCE', 
                eval=True, plt_imgs=None, scatter_plt=False, pltshow=False, output_dir='', save_model=False):
        train_data = dataset.train
        test_data = dataset.test
        # Data Loader for easy mini-batch return in training, the image batch shape will be (BATCH_SIZE, 1, 28, 28)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        if eval: 
            test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4)
        
        if opt=='Adam':
            # TODO: Investigate weight decay
            # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
          
        if loss=='BCE':     # TODO: Investigate BCE or MSE loss
            self.loss_func = nn.BCELoss()
        elif loss=='MSE':
            self.loss_func = nn.MSELoss()

        self.train()        # Set to train mode
        for epoch in range(epochs+1):
            epoch_loss = 0      # printing intermediary loss
            for batch_idx, (batch_train, _) in enumerate(train_loader):
                batch_train = batch_train.to(self.device)               # moving batch to GPU if available
                batch_x = batch_train.view(batch_train.size(0), -1)     # batch x, shape (batch, 28*28)
                batch_y = batch_train.view(batch_train.size(0), -1)     # batch y, shape (batch, 28*28)
                
                # =================== forward ===================== #
                encoded, decoded = self.forward(batch_x)
                loss = self.loss_func(decoded, batch_y)      
                MSE_loss = nn.MSELoss()(decoded, batch_y)   # mean square error
                # =================== backward ==================== #
                self.optimizer.zero_grad()               # clear gradients for this training step
                loss.backward()                          # backpropagation, compute gradients
                self.optimizer.step()                    # apply gradients

                epoch_loss += loss.data
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss:{:.6f} \t MSE Loss:{:.6f} '.format(
                        epoch, batch_idx * len(batch_train), len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.data / len(batch_train),
                            MSE_loss.data / len(batch_train)
                    ))
                    
            epoch_loss /= len(train_loader.dataset)
            print('\n====> Epoch: {} Average loss: {:.4f}'.format(epoch, epoch_loss))
        
            # =================== EVAL MODEL ==================== #
            if eval:
                self.eval()        # set dropout and batch normalisation layers to eval mode
                test_loss = 0   
                feat_total = []
                target_total = []

                with torch.no_grad():      # turn autograd off for memory efficiency
                    for batch_idx, (batch_test, batch_test_label) in enumerate(test_loader):
                        batch_test = batch_test.to(self.device)
                        batch_test = batch_test.view(batch_test.size(0), -1)  
                        encoded, decoded = self.forward(batch_test)

                        # Accumulate test loss, feat_total and target_total
                        test_loss += self.loss_func(decoded, batch_test.view(batch_test.size(0), -1)).data
                        feat_total.append(encoded.data.cpu().view(-1, encoded.data.shape[1]))
                        target_total.append(batch_test_label)

                    test_loss /= len(test_loader.dataset)
                    print('====> Test set loss: {:.4f}\n'.format(test_loss))

                # =================== PLOT COMPARISON ===================== #
                if plt_imgs!=None and epoch % plt_imgs[1] == 0:         #plt_imgs = (N_TEST_IMGS, plt_interval)
                    batch_test = batch_test.view(-1, 1, 28, 28)         # Reshape into (N_TEST_IMG, 1, 28, 28)
                    decoded = decoded.view(-1, 1, 28, 28)
                    comparison = torch.cat([batch_test[:plt_imgs[0]], decoded.view(-1, 1, 28, 28)[:plt_imgs[0]]])
                    output_dir = self._check_output_dir(output_dir)
                    save_image(comparison.data.cpu(),
                            output_dir+'/x_recon_{}.png'.format(epoch), nrow=plt_imgs[0])

                # =================== PLOT SCATTER ===================== #
                if scatter_plt!=None and epoch % scatter_plt[1] == 0:       # scatter_plt = ('method', plt_interval)
                    feat_total = torch.cat(feat_total, dim=0)
                    target_total = torch.cat(target_total, dim=0)
                    output_dir = self._check_output_dir(output_dir)
                    plt_scatter(feat_total.numpy(), target_total.numpy(), epoch, scatter_plt[0], output_dir, pltshow)


        # =================== SAVE MODEL AND DATA ==================== #
        if save_model: 
            output_dir = self._check_output_dir(output_dir)  
            save_path = '{}/{}.pth'.format(output_dir, output_dir.strip('./').strip('_output'))
            
            # Save dataset
            dataset.save_dataset(output_dir)
                
            # print(os.path.basename(os.path.normpath(save_path)))
            torch.save({        # Saving checkpt for inference and/or resuming training
                'model_name': os.path.basename(os.path.normpath(save_path)),
                'lr': lr,
                'batch_size': batch_size,
                'model_state_dict': self.state_dict(),
                'optimizer': self.optimizer,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss
                },
                save_path
            )
            
            config = {          # Save config file
                'model_name': os.path.basename(os.path.normpath(save_path)),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'lr': lr,
                'batch_size': batch_size,
                'optimizer': self.optimizer.__class__.__name__,
                'epoch': epoch,
                'loss': loss.data.item(),
                'loss_fn': self.loss_func.__class__.__name__
                }
            # print(config)

            with open(output_dir+'/config.json', 'w') as f:
                json.dump(config, f)

            print('\nAE Model saved to {}\n'.format(save_path))


    def load_model(self, output_dir):
        path = '{}/{}.pth'.format(output_dir, output_dir.strip('./').strip('_output'))
        # map the parameters from storage to location
        model_checkpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.model_name = model_checkpt['model_name'],
        self.lr = model_checkpt['lr'],
        self.batch_size = model_checkpt['batch_size'],
        self.load_state_dict(model_checkpt['model_state_dict'])
        self.optimizer = model_checkpt['optimizer']
        self.optimizer.load_state_dict(model_checkpt['optimizer_state_dict'])
        self.epoch = model_checkpt['epoch']
        self.loss = model_checkpt['loss']

        print('Loading model...\n{}\n'.format(self))
        print('Loaded model\t{}\n'.format(self.model_name))
        print('Batch size: {} LR: {} Optimiser: {}\n'
               .format(self.batch_size, self.lr, self.optimizer.__class__.__name__))
        print('Epoch: {}\tLoss: {}\n'      
                .format(self.epoch, self.loss))
        

    def _check_output_dir(self, output_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        return output_dir

    def _init_weights(self, layer):
        if type(layer) == nn.Linear:
            # aka Glorot initialisation (weight, gain('relu'))
            nn.init.xavier_uniform_(layer.weight, nn.init.calculate_gain('relu'))
            layer.bias.data.fill_(0.01)
