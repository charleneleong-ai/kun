#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Friday, August 30th 2019, 3:21:02 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Aug 30 2019
###

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
import os
import argparse
import glob
from datetime import datetime
import random

import torch
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns; sns.set()  # for plot styling

from scipy.stats import mode
from sklearn.metrics import accuracy_score
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import pairwise_distances_argmin_min

# from utils import plt_confusion_matrix
from utils.eval import cluster_accuracy  
from ae.ae import AutoEncoder
from ae.conv_ae import ConvAutoEncoder
from utils.datasets import FilteredMNIST
from utils.plt import plt_clusters

# Create output folder corresponding to current filename
CURRENT_FNAME = os.path.basename(__file__).split('.')[0]
timestamp = datetime.now().strftime('%Y.%d.%m-%H:%M:%S')
OUTPUT_DIR = './{}_{}_output'.format(CURRENT_FNAME, timestamp)

# Hyper Parameters
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3       
N_TEST_IMGS = 8

SEED = 489
torch.manual_seed(SEED)    # reproducible
np.random.seed(SEED)
random.seed(SEED)


if __name__ == '__main__':
    # Defaults set as vars above but can be overwritten by cmd line args
    parser = argparse.ArgumentParser(description='AE MNIST Example')
    parser.add_argument('--lr', type=float, default=LR, metavar='N',
                        help='learning rate for training (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--update_interval', type=int, default=100, metavar='N',
                        help='update interval for each batch')
    parser.add_argument('--epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train (default: 0)')
    parser.add_argument('--output', type=str, default='', metavar='N',
                        help='path/to/output/dir or latest')
    parser.add_argument('--label', type=int, default=1, metavar='N',
                        help='class to filter')
    args = parser.parse_args()
    

    if args.output=='':
        comment = '_lr={}_bs={}'.format(args.lr, args.batch_size) # Adding comment
        log_dir_name = OUTPUT_DIR.split('/')[1].split('_output')[0]+comment
        tb = SummaryWriter(log_dir='../runs/'+log_dir_name)    # Tensorboard
        
        # ae = ConvAutoEncoder()  
        ae = AutoEncoder(tb)     
    
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

    else:       # Load model and perform GMM
        OUTPUT_DIR = args.output   
        if args.output=='latest':
            OUTPUT_DIR = max(glob.iglob('./*/'), key=os.path.getctime)

        ae = AutoEncoder()  
        dataset = FilteredMNIST(output_dir=OUTPUT_DIR)
        model_path = ae.load_model(output_dir=OUTPUT_DIR)

        if args.epochs != 0:    # Further training if needed
            ae.fit(dataset, 
                batch_size=ae.BATCH_SIZE, 
                epochs=args.epochs, 
                lr=ae.LR, 
                opt=ae.optimizer.__class__.__name__,         # Adam
                loss=float(ae.loss),         # BCE or MSE
                patience=3,        # Num epochs for early stopping
                eval=True,          # Eval training process with test data
                plt_imgs=(N_TEST_IMGS, 10),         # (N_TEST_IMGS, plt_interval)
                scatter_plt=('tsne', 10),           # ('method', plt_interval)
                output_dir=OUTPUT_DIR, 
                save_model=True)        # Update old run
        
        _, feat, labels, test_imgs = ae.eval_model(dataset=dataset, 
                                batch_size=ae.BATCH_SIZE, 
                                epoch=ae.EPOCH, 
                                plt_imgs=None, 
                                # scatter_plt=('tsne', ae.EPOCHS),    
                                output_dir=OUTPUT_DIR)

        print(feat.size())
        print(dataset.test.targets.unique()) 
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=SEED)
        feat = tsne.fit_transform(feat.numpy())
        # pca = PCA(n_components=2, random_state=SEED)
        # feat = pca.fit_transform(feat)
        print(feat.shape)

        feat = StandardScaler().fit_transform(feat)    # Normalise the data
        # n_components = np.arange(1, 21)
        # models = [GaussianMixture(n, covariance_type='full', random_state=SEED).fit(feat) for n in n_components]
        # bic = [m.bic(feat) for m in models]
        # aic = [m.aic(feat) for m in models]

        # ymin = min(bic)     # Finding min pt
        # xmin = bic.index(ymin)
        # k = n_components[xmin]
        # print(k, ' components')

        # fig = plt.figure()
        # plt.plot(n_components, bic, label='BIC')
        # plt.plot(n_components, aic,  label='AIC')
        # plt.legend(loc='best')
        # plt.xlabel('n_components')
        # plt.savefig(OUTPUT_DIR+'/optimal_k.png', bbox_inches='tight')
        # plt.close(fig)
      
        # ae.tb.add_image(tag='optimal_k.png', 
        #                 img_tensor=plt.imread(OUTPUT_DIR+'/optimal_k.png'), 
        #                 global_step=ae.EPOCH, dataformats='HWC')
        
        OUTPUT_DIR = OUTPUT_DIR+'_tsne'
        
        # plt_img = plt_clusters(OUTPUT_DIR+'_HDBSCAN.png',feat, hdbscan.HDBSCAN, (), {'min_cluster_size':100}) 
        # ae.tb.add_image(tag='_tsne_HDBSCAN.png', 
        #                 img_tensor=plt_img, 
        #                 global_step = ae.EPOCH, dataformats='HWC')

        hdbscan = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
        hdbscan.fit(feat)
    
        #hdbscan.minimum_spanning_tree_.plot(edge_cmap='viridis', 
        #                                     edge_alpha=0.6, 
        #                                     node_size=10, 
        #                                     edge_linewidth=1)
        # plt.savefig(OUTPUT_DIR+'_HDBSCAN_min_span_tree.png', bbox_inches='tight')
        # plt.close()
        # ae.tb.add_image(tag='_HDBSCAN_min_span_tree.png', 
        #                         img_tensor=plt.imread(OUTPUT_DIR+'_HDBSCAN_min_span_tree.png'), 
        #                         global_step = ae.EPOCH, dataformats='HWC')

        #hdbscan.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        # plt.savefig(OUTPUT_DIR+'_HDBSCAN_single_linkage_tree.png', bbox_inches='tight')
        # plt.close()
        # ae.tb.add_image(tag='_HDBSCAN_single_linkage_tree.png', 
        #                                 img_tensor=plt.imread(OUTPUT_DIR+'_HDBSCAN_single_linkage_tree.png'), 
        #                                 global_step = ae.EPOCH, dataformats='HWC')

        #hdbscan.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette('hls'))
        # plt.savefig(OUTPUT_DIR+'_HDBSCAN_condensed_tree.png', bbox_inches='tight')
        # plt.close()
        # ae.tb.add_image(tag='_HDBSCAN_condensed_tree.png', 
        #                                 img_tensor=plt.imread(OUTPUT_DIR+'_HDBSCAN_condensed_tree.png'), 
        #                                 global_step = ae.EPOCH, dataformats='HWC')
        labels = hdbscan.labels_
        print(np.unique(labels), labels.shape)

        palette = sns.color_palette('hls', np.unique(labels).max() +1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]    # -1 is noise
        ax = plt.subplot()
        ax.tick_params(axis='both', labelsize=10)
        plt.scatter(feat.T[0], feat.T[1], c=colors, s=8, linewidths=1)

        centroids = np.array([np.median(feat[labels == label, :], axis=0) for label in np.unique(labels)[1:]])
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, alpha=0.5)
        
        plt.savefig(OUTPUT_DIR+'_HDBSCAN_clusters_{}.png'.format(ae.EPOCH), bbox_inches='tight')
        plt.close()
        ae.tb.add_image(tag='_HDBSCAN_clusters.png', 
                        img_tensor=plt.imread(OUTPUT_DIR+'_HDBSCAN_clusters.png'), 
                        global_step = ae.EPOCH, dataformats='HWC')

        closest, _ = pairwise_distances_argmin_min(centroids, feat)
        test_imgs = test_imgs.view(-1, 1, 28, 28)
        print(closest, test_imgs.size())
        centroid_imgs = []
        for i in closest:
            save_image(test_imgs[i], OUTPUT_DIR+'closest_{}.png'.format(i))


        



