#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, August 22nd 2019, 9:25:01 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Wed Aug 28 2019
# -----
# Copyright (c) 2019 Victoria University of Wellington ECS
###

import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn.metrics

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
sns.set(font_scale=2)

SEED = 489

def plt_scatter(feat, labels, epoch, method, output_dir, pltshow=False):
    print('Plotting scatter_{}_{}.png \n'.format(method, epoch))
    
    if feat.shape[1] > 2:   # Plot only first 5000 pts
        if feat.shape[0] > 5000:
            feat = feat[:5000, :]
            labels = labels[:5000]

        if method == 'pca':
            feat = pca(feat)
        elif method == 'tsne':
            feat = tsne(feat)

    labels_list = np.unique(labels)
    
    plt.ion()
    plt.clf()
    palette = np.array(sns.color_palette('hls', len(labels_list)))
    ax = plt.subplot(aspect='equal')
    
    for i, label in enumerate(labels_list):
        plt.plot(feat[labels == label, 0], feat[labels == label, 1], '.', c=palette[i])
    # plt.legend(loc='upper right')
    # plt.legend(dataset.targets.unique().tolist().sort(), loc='upper right', prop={'size': 6})
        
        ax.axis('tight')
        xtext, ytext = np.median(feat[labels == label, :], axis=0)
        txt = ax.text(xtext, ytext, str(label), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])

    plt.draw()
    plt.ioff()
    plt.savefig(output_dir+'/scatter_{}_{}.png'.format(method, epoch), bbox_inches='tight')

    if pltshow:
        plt.show()
        
def plt_confusion_matrix(y_pred, y_target, output_dir, pltshow=False):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_target, y_pred)

    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
    # plt.title("Confusion matrix", fontsize=10)
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Clustering label', fontsize=20)
    plt.savefig(output_dir+'/confusion_matrix.png', bbox_inches='tight')

    if pltshow:
        plt.show()


def tsne(feat):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=SEED)
    return tsne.fit_transform(feat)

def pca(feat):
    pca = PCA(n_components=2, random_state=SEED)
    return pca.fit_transform(feat)


    