#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 4:18:39 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sun Sep 15 2019
###
from datetime import datetime

from server.model.ae import AutoEncoder
from server.utils.filtered_MNIST import FilteredMNIST

def train(dataset):
    EPOCHS = 50
    BATCH_SIZE = 128
    LR = 1e-3       
    N_TEST_IMGS = 8

    ae = AutoEncoder()

    timestamp = datetime.now().strftime('%Y.%d.%m-%H:%M:%S')
    OUTPUT_DIR = './output/{}_{}_{}'.format('ae', dataset.LABEL, timestamp)
    print(OUTPUT_DIR)
    ae.fit(dataset, 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            lr=LR, 
            opt='Adam',         # Adam
            loss='BCE',         # BCE or MSE
            patience=10,        # Num epochs for early stopping
            eval=True,          # Eval training process with test data
            plt_imgs=(N_TEST_IMGS, 10),         # (N_TEST_IMGS, plt_interval)
            scatter_plt=('tsne', 10),           # ('method', plt_interval)
            output_dir=OUTPUT_DIR, 
            save_model=True)        # Also saves dataset
    
    return True

def filtered_MNIST(label):
    return FilteredMNIST(label=label, split=0.8, n_noise_clusters=3)


