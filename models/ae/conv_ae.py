#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Tuesday, August 27th 2019, 6:34:04 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Wed Aug 28 2019
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

CURRENT_FNAME = os.path.basename(__file__).split('.')[0]

class ConvAutoEncoder(nn.Module):
    def __init__(self, ):
        super(ConvAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # conv layer (depth from 1 --> 16), 3x3 kernels
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(inplace=True),  # modify input directly     
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            # conv layer (depth from 16 --> 8), 3x3 kernels
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        
        # at this point the representation is (8, 2, 2) i.e. 32-dim

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
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
    
    def fit(self, dataset, batch_size, epochs, lr, opt='Adam', loss='MSE', 
                eval=True, plt_imgs=None, scatter_plt=False, pltshow=False, output_dir='', save_model=False):
        train_data = dataset.train
        test_data = dataset.test

        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LR = lr
        self.OUTPUT_DIR = output_dir
        
        # Data Loader for easy mini-batch return in training, the image batch shape will be (BATCH_SIZE, 1, 28, 28)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
        if eval: 
            test_loader = Data.DataLoader(dataset=test_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
        
        if opt=='Adam':
            # TODO: Investigate weight decay
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.LR, weight_decay=1e-5)
            # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)
          
        if loss=='BCE':     # TODO: Investigate BCE or MSE loss
            self.loss_fn = nn.BCELoss()
        elif loss=='MSE':
            self.loss_fn = nn.MSELoss()

        self.train()        # Set to train mode
        for epoch in range(epochs+1):
            epoch_loss = 0      # printing intermediary loss
            for batch_idx, (batch_train, _) in enumerate(train_loader):
                batch_train = batch_train.to(self.device)               # moving batch to GPU if available

                # =================== forward ===================== #
                encoded, decoded = self.forward(batch_train)
                self.loss = self.loss_fn(decoded, batch_train)      
                MSE_loss = nn.MSELoss()(decoded, batch_train)   # mean square error
                # =================== backward ==================== #
                self.optimizer.zero_grad()               # clear gradients for this training step
                self.loss.backward()                     # backpropagation, compute gradients
                self.optimizer.step()                    # apply gradients

                epoch_loss += self.loss.item()*batch_train.size(0)
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss:{:.6f} \t MSE Loss:{:.6f} '.format(
                        epoch, batch_idx * len(batch_train), len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            self.loss.data / len(batch_train),
                            MSE_loss.data / len(batch_train)
                    ))
                    
            epoch_loss /= len(train_loader.dataset)
            print('\n====> Epoch: {} Average loss: {:.4f}'.format(epoch, epoch_loss))
        
            # =================== EVAL MODEL ==================== #
            if eval:
                self.eval_model(dataset, self.BATCH_SIZE, epoch, plt_imgs, scatter_plt, pltshow, self.OUTPUT_DIR)

        # =================== SAVE MODEL AND DATA ==================== #
        if save_model: 
            self.save_model(dataset, self.OUTPUT_DIR)
    
    def save_model(self, dataset, output_dir):
        output_dir = self._check_output_dir(output_dir)  
        save_path = '{}/{}.pth'.format(output_dir, output_dir.strip('./').strip('_output'))
        
        # Save dataset
        dataset.save_dataset(output_dir)
            
        # print(os.path.basename(os.path.normpath(save_path)))
        torch.save({        # Saving checkpt for inference and/or resuming training
            'model_name': os.path.basename(os.path.normpath(save_path)),
            'model_type': CURRENT_FNAME.split('.')[0],
            'model_state_dict': self.state_dict(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'lr': self.LR,
            'batch_size': self.BATCH_SIZE,
            'optimizer': self.optimizer,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epochs': self.EPOCHS,
            'loss': self.loss,
            'loss_fn': self.loss_fn
            },
            save_path
        )
        
        config = {          # Save config file
            'model_name': os.path.basename(os.path.normpath(save_path)),
            'model_type': CURRENT_FNAME.split('.')[0],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'lr': self.LR,
            'batch_size': self.BATCH_SIZE,
            'optimizer': self.optimizer.__class__.__name__,
            'epochs': self.EPOCHS,
            'loss': self.loss.data.item(),
            'loss_fn': self.loss_fn.__class__.__name__
            }
        # print(config)

        with open(output_dir+'/config.json', 'w') as f:
            json.dump(config, f)

        print('\nConv AE Model saved to {}\n'.format(save_path))


    def load_model(self, output_dir):
        path = '{}/{}.pth'.format(output_dir, output_dir.strip('./').strip('_output'))
        # map the parameters from storage to location
        model_checkpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.model_name = model_checkpt['model_name'],
        self.LR = model_checkpt['lr'],
        self.BATCH_SIZE = model_checkpt['batch_size'],
        self.load_state_dict(model_checkpt['model_state_dict'])
        self.optimizer = model_checkpt['optimizer']
        self.optimizer.load_state_dict(model_checkpt['optimizer_state_dict'])
        self.EPOCHS = model_checkpt['epochs']
        self.loss = model_checkpt['loss']
        self.loss_fn = model_checkpt['loss_fn']

        print('Loading model...\n{}\n'.format(self))
        print('Loaded model\t{}\n'.format(self.model_name))
        print('Batch size: {} LR: {} Optimiser: {}\n'
               .format(self.BATCH_SIZE, self.LR, self.optimizer.__class__.__name__))
        print('Epoch: {}\tLoss: {}\n'      
                .format(self.EPOCHS, self.loss))


    def eval_model(self, dataset, batch_size, epoch, plt_imgs=None, scatter_plt=False, pltshow=False, output_dir=''):
        
        test_loader = Data.DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=True, num_workers=4)

        self.eval()        # set dropout and batch normalisation layers to eval mode
        test_loss = 0   
        feat_total = []     # For TSNE
        target_total = []

        with torch.no_grad():      # turn autograd off for memory efficiency
            for batch_idx, (batch_test, batch_test_label) in enumerate(test_loader):
                batch_test = batch_test.to(self.device)
                encoded, decoded = self.forward(batch_test)

                loss = self.loss_fn(decoded, batch_test) 
                test_loss += loss.item()*batch_test.size(0)
                
                # Flatten for TSNE (b, 8, 2, 2) -> (b, 32)
                feat_total.append(encoded.data.cpu().view(batch_test.size(0), -1))   
                target_total.append(batch_test_label)

            test_loss /= len(test_loader.dataset)
            print('====> Test set loss: {:.4f}\n'.format(test_loss))

        # =================== PLOT COMPARISON ===================== #
        if plt_imgs!=None and epoch % plt_imgs[1] == 0:         # plt_imgs = (N_TEST_IMGS, plt_interval)
            batch_test = batch_test.view(-1, 1, 28, 28)         # Reshape into (N_TEST_IMG, 1, 28, 28)
            decoded = decoded.view(-1, 1, 28, 28)
            comparison = torch.cat([batch_test[:plt_imgs[0]], decoded.view(-1, 1, 28, 28)[:plt_imgs[0]]])
            
            output_dir = self._check_output_dir(output_dir)
            filename = 'x_recon_{}_{}.png'.format(CURRENT_FNAME.split('.')[0], epoch)
            print('Saving ', filename)
            save_image(comparison.data.cpu(), output_dir+'/'+filename, nrow=plt_imgs[0])

        # =================== PLOT SCATTER ===================== #
        if scatter_plt!=None and epoch % scatter_plt[1] == 0:       # scatter_plt = ('method', plt_interval)
            feat_total = torch.cat(feat_total, dim=0)
            target_total = torch.cat(target_total, dim=0)
            output_dir = self._check_output_dir(output_dir)
            plt_scatter(feat_total.numpy(), target_total.numpy(), epoch, scatter_plt[0], output_dir, pltshow)
     

    def _check_output_dir(self, output_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        return output_dir

    def _init_weights(self, layer):
        if type(layer) == nn.Linear:
            # aka Glorot initialisation (weight, gain('relu'))
            nn.init.xavier_uniform_(layer.weight, nn.init.calculate_gain('relu'))
            layer.bias.data.fill_(0.01)
