#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, August 22nd 2019, 9:25:01 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Aug 26 2019
# -----
# Copyright (c) 2019 Victoria University of Wellington ECS
###

import numpy as np
from scipy.optimize import linear_sum_assignment

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn.metrics

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
sns.set(font_scale=2)


def plt_scatter(feat, label, epoch, method, output_dir, pltshow=False):
    print('Plotting scatter_{}_{}.png \n'.format(method, epoch))
    
    if feat.shape[1] > 2:
        if feat.shape[0] > 5000:
            feat = feat[:5000, :]
            label = label[:5000]

        if method == 'pca':
            pca = PCA(n_components=2, random_state=489)
            feat = pca.fit_transform(feat)
        elif method == 'tsne':
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=489)
            feat = tsne.fit_transform(feat)

    plt.ion()
    plt.clf()
    palette = np.array(sns.color_palette('hls', 10))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.plot(feat[label == i, 0], feat[label == i, 1], '.', c=palette[i])
    # plt.legend(loc='upper right')
    #plt.legend(dataset.targets.unique().tolist().sort(), loc='upper right', prop={'size': 6})
    ax.axis('tight')
    for i in range(10):
        xtext, ytext = np.median(feat[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=18)
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


def cluster_accuracy(y_pred, y_target):
    """
    The problem of finding the best permutation to calculate the clustering accuracy 
    is a linear assignment problem.
    This function construct a N-by-N cost matrix
    """
    y_target = y_target.astype(np.int64)
    assert y_pred.size == y_target.size
    
    cluster_number = max(y_pred.max(), y_target.max()) + 1  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)   # Init count matrix
    for i in range(y_pred.size):
        count_matrix[y_pred[i], y_target[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_pred.size
    return accuracy, reassignment