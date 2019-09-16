#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 4:18:39 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Sep 16 2019
###
import os
import glob
from datetime import datetime

import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min
from hdbscan import HDBSCAN
from torchvision.utils import save_image, make_grid

from server import create_app
from server.model.ae import AutoEncoder
from server.model.som import SOM
from server.model.utils.plt import plt_scatter
from server.utils.filtered_MNIST import FilteredMNIST


# Import current app settings for tasks
app = create_app()
app.app_context().push()

SEED = 489
np.random.seed(SEED)
random.seed(SEED)

def filtered_MNIST(label):
    return FilteredMNIST(label=label, split=0.8, n_noise_clusters=3, download_dir=app.config['RAW_IMG_DIR'])
    
def train(dataset):
    EPOCHS = 1
    BATCH_SIZE = 128
    LR = 1e-3       
    N_TEST_IMGS = 8

    ae = AutoEncoder()

    timestamp = datetime.now().strftime('%Y.%d.%m-%H:%M:%S')
    MODEL_OUTPUT_DIR = app.config['MODEL_OUTPUT_DIR']
    OUTPUT_DIR = os.path.join(MODEL_OUTPUT_DIR, '{}_{}_{}'.format('ae', dataset.LABEL, timestamp))
    print(OUTPUT_DIR)
    ae.fit(dataset, 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            lr=LR, 
            opt='Adam',         # Adam
            loss='BCE',         # BCE or MSE
            patience=10,        # Num epochs for early stopping
            eval=True,          # Eval training process with test data
            # plt_imgs=(N_TEST_IMGS, 10),         # (N_TEST_IMGS, plt_interval)
            # scatter_plt=('tsne', 10),           # ('method', plt_interval)
            output_dir=OUTPUT_DIR, 
            save_model=True)        # Also saves dataset
    
    return OUTPUT_DIR

def process_imgs():
    # Return latest model by default
    OUTPUT_DIR = max(glob.iglob(app.config['MODEL_OUTPUT_DIR']+'/*/'), key=os.path.getctime)
    ae, feat_ae, labels, imgs = load_model(OUTPUT_DIR)

    print('Clustering with HDBSCAN...\n')
    feat = tsne(feat_ae, 2)     # HDBSCAN works best with 2dim w/ small dataset
    feat = MinMaxScaler().fit_transform(feat) 
    c_labels = hdbscan(feat, min_cluster_size=10)   
    c_labels = order_c_labels(c_labels)
    
    # img_plt = plt_scatter(feat, c_labels, output_dir=OUTPUT_DIR, plt_name='_{}.png'.format('hdbscan'), pltshow=False)
    # ae.tb.add_image(tag='_{}'.format('hdbscan'), 
    #                 img_tensor=img_plt, 
    #                 global_step = ae.EPOCH, dataformats='HWC')

    print('Ordering image grid with Self Organising Map...\n')
    label_0_idx = np.where(c_labels==0)[0]  # Assume largest cluster is img label
    lut = dict(enumerate(list(label_0_idx)))
    # Sample 3D feat for better cluster seperation in SOM
    data = feat_ae[label_0_idx].numpy()
    data = tsne(data, 3)
    
    iter = 1000
    som = SOM(data=data, dims=[20, 10], n_iter = iter, lr=0.01)
    net = som.train()
    print(net.shape)
    net_w = np.array([net[x-1, y-1, :] for x in range(net.shape[0]) for y in range(net.shape[1])])
    print(net_w.shape)
    
    # This is producing duplicate
    img_grd_idx, _ = pairwise_distances_argmin_min(net_w, data)
    print(img_grd_idx, img_grd_idx.shape)
    img_grd_idx = np.array([lut[i] for i in img_grd_idx])   # Remapping to label 0 idx
    print(img_grd_idx, img_grd_idx.shape)

    # Saving image grid scatter
    img_plt = plt_scatter([feat, feat[img_grd_idx]], c_labels, colors=['blue'], 
                            output_dir=OUTPUT_DIR, plt_name='_{}_som_3D_{}.png'.format('hdbscan', iter), pltshow=False)
    ae.tb.add_image(tag='_{}_som_3D_{}.png'.format('hdbscan', iter), 
                                        img_tensor=img_plt, 
                                        global_step = ae.EPOCH, dataformats='HWC')
    # Saving image grid
    img_grd = imgs[img_grd_idx]
    print(img_grd.size())
    img_grd = img_grd.view(-1, 1, 28, 28)
    print(img_grd.size())
    save_image(img_grd, OUTPUT_DIR+'_img_grd_som_3D_{}.png'.format(iter), nrow=20)
    ae.tb.add_images(tag='_img_grd_som_3D_{}.png'.format(iter), 
                            img_tensor=img_grd, 
                            global_step = ae.EPOCH)

    # Saving imgs to client imgs folders
    for i, img in enumerate(img_grd):
        save_image(img, app.config['IMG_DIR']+'/{}_{}.png'.format(i, img_grd_idx[i]))
    
                            
# Helper functions for process_imgs()
def load_model(output_dir):
    ae = AutoEncoder()  
    dataset = FilteredMNIST(output_dir=output_dir)
    model_path = ae.load_model(output_dir=output_dir)
    # dataset.test += dataset.train   # Get all the data, eval_model loads dataset.test
    _, feat, labels, imgs = ae.eval_model(dataset=dataset, output_dir=output_dir)
    return ae, feat, labels, imgs
    
def tsne(feat, dim):
    tsne = TSNE(perplexity=30, n_components=dim, init='pca', n_iter=1000, random_state=SEED)
    feat = tsne.fit_transform(feat)
    feat = MinMaxScaler().fit_transform(feat) 
    return feat

def hdbscan(feat, min_cluster_size):
    cluster = HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=False)
    cluster.fit(feat)
    return cluster.labels_

def order_c_labels(c_labels):
    cluster_size = [(label, len(c_labels[c_labels==label])) for label in np.unique(c_labels[c_labels!=-1])] # Tuple
    sorted_cluster_size = sorted(cluster_size, key=lambda x:x[1])[::-1] # Sort by cluster_size[1] large to small
    sorted_cluster_size = [c[0] for c in sorted_cluster_size]   # Return idx
    lut = np.array([i[0] for i in sorted(enumerate(sorted_cluster_size), key=lambda x:x[1])])
    c_labels[c_labels!=-1] = lut[c_labels[c_labels!=-1]]    # Keep noise c_labels
    return c_labels