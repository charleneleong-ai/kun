#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, September 5th 2019, 3:19:07 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Thu Sep 05 2019
###


import os
import glob
import warnings
warnings.filterwarnings('ignore')
import sys
import argparse

import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from torchvision.utils import save_image, make_grid

from hdbscan import HDBSCAN

from ae.ae import AutoEncoder
from ae.convae import ConvAutoEncoder
from utils.datasets import FilteredMNIST
from utils.plt import plt_scatter

SEED=489
np.random.seed(SEED)
random.seed(SEED)

GRID_SIZE = 200

# Return latest model by default
OUTPUT_DIR = max(glob.iglob('./output/*output/'), key=os.path.getctime)
print(OUTPUT_DIR)

def load_model(model):
    if model == 'ae':
        ae = AutoEncoder()  
        
    dataset = FilteredMNIST(output_dir=OUTPUT_DIR)
    model_path = ae.load_model(output_dir=OUTPUT_DIR)
    # dataset.test += dataset.train   # Get all the data, eval_model loads dataset.test
    _, feat, labels, imgs = ae.eval_model(dataset=dataset, output_dir=OUTPUT_DIR)
    
    return ae, feat, labels, imgs

def process_feat(feat):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=SEED)
    feat = tsne.fit_transform(feat.numpy())
    feat = MinMaxScaler().fit_transform(feat) 
    return feat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster AE MNIST Example')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, metavar='N',
                        help='path/to/output/dir (default: latest)')
    parser.add_argument('--cluster', type=str, default='hdbscan', metavar='N',
                        help='clustering alg (default: hdbscan)')
    args = parser.parse_args()

    OUTPUT_DIR = args.output

    model = OUTPUT_DIR.split('_')[1]
    ae, feat, labels, imgs = load_model(model)
    feat = process_feat(feat)
    print(feat.shape)

    
    if args.cluster == 'hdbscan':
        cluster = HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
        cluster.fit(feat)
        c_labels = cluster.labels_

    print(np.unique(c_labels), c_labels.shape)
    
    # =================== ORDER CLUSTER ==================== #
    # Ordering cluster from largest to smallest [0, 1, 2...]
    cluster_size = [(label, len(c_labels[c_labels==label])) for label in np.unique(c_labels[c_labels!=-1])] # Tuple
    sorted_cluster_size = sorted(cluster_size, key=lambda x:x[1])[::-1] # Sort by cluster_size[1] large to small
    sorted_cluster_size = [c[0] for c in sorted_cluster_size]   # Return idx
    # print(cluster_size)
    # print(sorted_cluster_size)
    # Get lut idx of sorted array
    lut = np.array([i[0] for i in sorted(enumerate(sorted_cluster_size), key=lambda x:x[1])])
    # print(lut)

    
    c_labels[c_labels!=-1] = lut[c_labels[c_labels!=-1]]    # Keep noise c_labels
    
    img_plt = plt_scatter(feat, c_labels, output_dir=OUTPUT_DIR, plt_name='_{}.png'.format(args.cluster), pltshow=False)
    ae.tb.add_image(tag='_'+args.cluster, 
                    img_tensor=img_plt, 
                    global_step = ae.EPOCH, dataformats='HWC')


    # =================== CLEAN LABEL ==================== #
    # Clean largest cluster for true label
    label_0_idx = np.where(c_labels==0)[0]
    sample_idx = random.sample(set(label_0_idx), GRID_SIZE)
    
    img_plt = plt_scatter([feat, feat[sample_idx]], c_labels, ['blue'], 
                        output_dir=OUTPUT_DIR, plt_name='_{}_sample.png'.format(args.cluster), pltshow=False)
    ae.tb.add_image(tag='_{}_sample'.format(args.cluster), 
                        img_tensor=img_plt, 
                        global_step = ae.EPOCH, dataformats='HWC')

    # Plot image grid
    image_grid = imgs[sample_idx]
    print(image_grid.size())
    image_grid = image_grid.view(-1, 1, 28, 28)
    print(image_grid.size())
    save_image(image_grid, OUTPUT_DIR+'_image_grid.png', nrow=20)
    
    
    # Checking sample idx row by row
    sample_idx = np.array(sample_idx).reshape(-1, 20)
    image_grid_0 = imgs[sample_idx[1]].view(-1, 1, 28, 28)
    save_image(image_grid_0, OUTPUT_DIR+'_image_grid_0.png', nrow=20)