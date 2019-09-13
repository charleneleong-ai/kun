#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, September 5th 2019, 3:19:07 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Sep 13 2019
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
import somoclu

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
    feat = tsne(feat_ae, 3)
    feat = MinMaxScaler().fit_transform(feat) 
    print(feat.shape)

    if args.cluster == 'hdbscan':
        cluster = HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
        cluster.fit(feat)
        c_labels = cluster.labels_

    print(np.unique(c_labels), c_labels.shape)

    plt_scatter_3D(feat, labels, output_dir=OUTPUT_DIR, plt_name='tsne_{}_3D.png'.format(ae.EPOCH), pltshow=False)

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
    
    #img_plt = plt_scatter(feat, c_labels, output_dir=OUTPUT_DIR, plt_name='_{}.png'.format(args.cluster), pltshow=False)
    # ae.tb.add_image(tag='_'+args.cluster, 
    #                 img_tensor=img_plt, 
    #                 global_step = ae.EPOCH, dataformats='HWC')

    plt_scatter_3D(feat, c_labels, output_dir=OUTPUT_DIR, plt_name='_{}_3D.png'.format(args.cluster), pltshow=False)


    # =================== CLEAN LABEL ==================== #
    # Clean largest cluster for true label
    label_0_idx = np.where(c_labels==0)[0]
    sample_idx = random.sample(set(label_0_idx), GRID_SIZE)
    
    # img_plt = plt_scatter([feat, feat[sample_idx]], c_labels, ['blue'], 
    #                     output_dir=OUTPUT_DIR, plt_name='_{}_sample.png'.format(args.cluster), pltshow=False)
    # # # ae.tb.add_image(tag='_{}_sample'.format(args.cluster), 
    # #                     img_tensor=img_plt, 
    # #                     global_step = ae.EPOCH, dataformats='HWC')
    
    # Plot image grid
    # image_grid = imgs[sample_idx]
    # print(image_grid.size())
    # image_grid = image_grid.view(-1, 1, 28, 28)
    # print(image_grid.size())
    # save_image(image_grid, OUTPUT_DIR+'_image_grid_sample.png', nrow=20)

    # =================== SOM ==================== #
    # lut = dict(enumerate(list(label_0_idx)))
    
    # data = feat_ae[label_0_idx]
    # data = tsne(data, 3)
    # # data = feat[sample_idx]
    # print(data.shape)
    # for iter in [6000, 7000, 8000, 9000, 10000]:  
    #     som = SOM(data=data, dims=[20, 10], n_iter = iter, lr=0.01)
    #     net = som.train()

    #     centroids = np.array([net[x-1, y-1, :] for x in range(net.shape[0]) for y in range(net.shape[1])])

    #     # data = tsne(data)
    #     # centroids = tsne(centroids)
    #     data_sample = data

    #     img_grd_idx, img_dist = pairwise_distances_argmin_min(centroids, data)
    #     # print(np.unique(img_grd_idx))
    #     print(img_grd_idx.reshape(20,10))
    #     print(img_dist.reshape(20,10))
    #     # img_grd_idx = np.array(img_grd_idx)
    #     # print(img_grd_idx.reshape(20,10), min(img_grd_idx), max(img_grd_idx))

    #     img_grd_idx = np.array([lut[i] for i in img_grd_idx])
    #     # print(img_grd_idx.reshape(20,10))
        
    #     img_plt = plt_scatter([feat, feat[img_grd_idx]], c_labels, ['blue'], 
    #                         output_dir=OUTPUT_DIR, plt_name='_{}_som_3D_{}.png'.format(args.cluster, iter), pltshow=False)

        
    #     image_grid = imgs[img_grd_idx]
    #     print(image_grid.size())
    #     image_grid = image_grid.view(-1, 1, 28, 28)
    #     print(image_grid.size())
    #     save_image(image_grid, OUTPUT_DIR+'_image_grid_som_3D_{}.png'.format(iter), nrow=20)



    
    
    # # Checking sample idx row by row
    # sample_idx = np.array(sample_idx).reshape(-1, 20)
    # image_grid_0 = imgs[sample_idx[1]].view(-1, 1, 28, 28)
    # save_image(image_grid_0, OUTPUT_DIR+'_image_grid_0.png', nrow=20)

# if __name__ == '__main__':
#     n_iter = 2000

#     data = np.random.randint(0, 255, (3, 2000))
#     data = data / data.max()
#     som = SOM(data=data, dims=[20, 10], n_iter = n_iter, lr=0.01)
#     net = som.train()

    # fig = plt.figure()

    # ax = fig.add_subplot(111, aspect='equal')
    # ax.set_xlim((0, net.shape[0]+1))
    # ax.set_ylim((0, net.shape[1]+1))
    # ax.set_title('Self-Organising Map after %d iterations' % n_iter)

    # # plot rectangles
    # for x in range(1, net.shape[0] + 1):
    #     for y in range(1, net.shape[1] + 1):
    #         ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
    #                     facecolor=net[x-1,y-1,:],
    #                     edgecolor='none'))
    # plt.show()