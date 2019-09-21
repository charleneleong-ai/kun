#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, September 5th 2019, 3:19:07 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sat Sep 21 2019
###


import os
import glob
import warnings
warnings.filterwarnings('ignore')
import sys
print(sys.executable)
import argparse

import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min

from torchvision.utils import save_image, make_grid

from hdbscan import HDBSCAN

from ae import AutoEncoder, ConvAutoEncoder
from utils.datasets import FilteredMNIST
from utils.plt import plt_scatter, plt_scatter_3D
from som import SOM
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import patches as patches


SEED=489
np.random.seed(SEED)
random.seed(SEED)

GRID_SIZE = 200

# Return latest model by default
OUTPUT_DIR = max(glob.iglob('./output/train_ae*/'), key=os.path.getctime)
print(OUTPUT_DIR)

def load_model(model):
    if model == 'ae':
        ae = AutoEncoder()  
        
    dataset = FilteredMNIST(output_dir=OUTPUT_DIR)
    model_path = ae.load_model(output_dir=OUTPUT_DIR)
    # dataset.test += dataset.train   # Get all the data, eval_model loads dataset.test
    _, feat, labels, imgs = ae.eval_model(dataset=dataset, output_dir=OUTPUT_DIR)
    
    return ae, feat, labels, imgs


def tsne(feat, dim):
    tsne = TSNE(perplexity=30, n_components=dim, init='pca', n_iter=1000, random_state=SEED)
    feat = tsne.fit_transform(feat)
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
    ae, feat_ae, labels, imgs = load_model(model)
    feat = tsne(feat_ae, 2)
    feat = MinMaxScaler().fit_transform(feat) 
    print(feat.shape)

    if args.cluster == 'hdbscan':
        cluster = HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
        cluster.fit(feat)
        c_labels = cluster.labels_

    print(np.unique(c_labels), c_labels.shape)

    # plt_scatter_3D(feat, labels, output_dir=OUTPUT_DIR, plt_name='tsne_{}_3D.png'.format(ae.EPOCH), pltshow=False)

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

    # plt_scatter_3D(feat, c_labels, output_dir=OUTPUT_DIR, plt_name='_{}_3D.png'.format(args.cluster), pltshow=False)


    # =================== SAMPLE CLUSTER ==================== #
    # Assume largest cluster is img label
    label_0_idx = np.where(c_labels==0)[0]
    sample_idx = random.sample(set(label_0_idx), GRID_SIZE)
    
    # img_plt = plt_scatter([feat, feat[sample_idx]], c_labels, ['blue'], 
    #                     output_dir=OUTPUT_DIR, plt_name='_{}_sample.png'.format(args.cluster), pltshow=False)
    # ae.tb.add_image(tag='_{}_sample'.format(args.cluster), 
    #                     img_tensor=img_plt, 
    #                     global_step = ae.EPOCH, dataformats='HWC')
    

    # =================== ORDER CLUSTER ==================== #
    # using self organising map (SOM)
    lut = dict(enumerate(list(label_0_idx)))
    
    # Sample 3D feat for better cluster seperation
    # data = feat_ae[label_0_idx].numpy()
    # data = tsne(data, 3)
    data = feat[label_0_idx]
    print(data.shape)
    for iter in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:  
        for lr in [0.01, 0.1, 0.2, 0.5, 0.8, 1]:
            som = SOM(data=data, dims=[10, 20], n_iter = iter, lr=lr)
            # ae.tb.add_scalar('LR', lr, self.EPOCH)
            net = som.train()
            print(net.shape)
            net_w = np.array([net[x-1, y-1, :] for x in range(net.shape[0]) for y in range(net.shape[1])])
            print(net_w.shape)
            
            # This is producing duplicate
            img_grd_idx, _ = pairwise_distances_argmin_min(net_w, data)
            print(img_grd_idx, img_grd_idx.shape)

            img_grd_idx = np.array([lut[i] for i in img_grd_idx])
            print(img_grd_idx, img_grd_idx.shape)

            # data_sample = data
            # mindist = np.array([min(net_w, key=lambda p: np.sqrt(sum((p - c)**2))) for c in net_w])
            # # print(mindist, mindist.shape)
            # mindist_idx = np.where(mindist==net_w)[0]
            # # print(mindist_idx, mindist_idx.shape)
            # mindist_idx = np.unique(mindist_idx)
            # mindist_idx = np.array([lut[i] for i in mindist_idx]
            
            # img_plt = plt_scatter([feat, feat[mindist_idx]], c_labels, colors=['blue'], 
            #                     output_dir=OUTPUT_DIR, plt_name='mindist/_{}_som_3D_{}.png'.format(args.cluster, iter), pltshow=False)

            # img_grd = imgs[mindist_idx]
            # print(img_grd.size())
            # img_grd = img_grd.view(-1, 1, 28, 28)
            # print(img_grd.size())
            # save_image(img_grd, OUTPUT_DIR+'mindist/_img_grd_som_3D_{}.png'.format(iter), nrow=20)
            
            img_plt = plt_scatter([feat, feat[img_grd_idx]], c_labels, colors=['blue'], 
                                output_dir=OUTPUT_DIR, plt_name='_{}_som_2D_{}_lr={}.png'.format(args.cluster, iter, lr), pltshow=False)
            ae.tb.add_image(tag='_{}_som_3D_{}.png'.format(args.cluster, iter), 
                                            img_tensor=img_plt, 
                                            global_step = ae.EPOCH, dataformats='HWC')

            img_grd = imgs[img_grd_idx]
            print(img_grd.size())
            img_grd = img_grd.view(-1, 1, 28, 28)
            print(img_grd.size())
            save_image(img_grd, OUTPUT_DIR+'_img_grd_som_2D_{}_lr={}.png'.format(iter, lr), nrow=20)
            ae.tb.add_images(tag='_img_grd_som_2D_{}_lr={}.png'.format(iter, lr), 
                                    img_tensor=make_grid(img_grd, nrow=20), 
                                    global_step = ae.EPOCH)
