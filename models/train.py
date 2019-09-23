#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, September 5th 2019, 2:25:54 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Sep 23 2019
###

import sys
import os
import argparse
import glob
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from ae.ae import AutoEncoder
from ae.convae import ConvAutoEncoder
from utils.datasets import FilteredMNIST


# Create output folder corresponding to current filename
CURRENT_FNAME = os.path.basename(__file__).split('.')[0]
timestamp = datetime.now().strftime('%Y.%d.%m-%H:%M:%S')

# Hyper Parameters
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3       
N_TEST_IMGS = 8


if __name__ == '__main__':
    # Defaults set as vars above but can be overwritten by cmd line args
    parser = argparse.ArgumentParser(description='Train AE MNIST Example')
    parser.add_argument('--lr', type=float, default=LR, metavar='N',
                        help='learning rate for training (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--model', type=str, default='ae', metavar='N',
                        help='type of model ')
    parser.add_argument('--label', type=int, default=8, metavar='N',
                        help='class to filter')
    args = parser.parse_args()
    
    
    # ./output/train_ae_[label]_[timestamp]_output/
    OUTPUT_DIR = './output/{}_{}_{}_{}_output'.format(CURRENT_FNAME, args.model, args.label, timestamp)

    # comment = '_lr={}_bs={}'.format(args.lr, args.batch_size) # Adding comment
    # log_dir_name = os.path.basename(os.path.normpath(OUTPUT_DIR)).replace('_output','')+comment
    # print(log_dir_name)
    # tb = SummaryWriter(log_dir='./tb_runs/'+log_dir_name)    # Tensorboard

    if args.model=='ae':
        ae = AutoEncoder() 
    elif args.model=='conv_ae':
        ae = ConvAutoEncoder() 

    dataset = FilteredMNIST(label=args.label, split=0.8, n_noise_clusters=3)

    print(dataset.train.targets.unique(), len(dataset.train), len(dataset.test))
    
    ae.fit(dataset, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            lr=args.lr, 
            opt='Adam',         # Adam
            loss='BCE',         # BCE or MSE
            patience=10,        # Num epochs for early stopping
            eval=True,          # Eval training process with test data
            plt_imgs=(N_TEST_IMGS, 10),         # (N_TEST_IMGS, plt_interval)
            scatter_plt=('tsne', 10),           # ('method', plt_interval)
            output_dir=OUTPUT_DIR, 
            save_model=True)        # Also saves dataset
