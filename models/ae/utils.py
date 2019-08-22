#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, August 22nd 2019, 9:25:01 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Thu Aug 22 2019
# -----
# Copyright (c) 2019 Victoria University of Wellington ECS
###

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns



def scatter(feat, label, epoch, method, output_dir):
    print('Plotting scatter_{}_{}.png \n'.format(method, epoch))
    if feat.shape[1] > 2:
        if feat.shape[0] > 5000:
            feat = feat[:5000, :]
            label = label[:5000]

        if method == 'pca':
            pca = PCA(n_components=2).fit(feat)
            feat = pca.transform(feat)
        elif method == 'tsne':
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
            feat = tsne.fit_transform(feat)
        else:
            print('Invalid dim reduction method.')
            return

    plt.ion()
    plt.clf()
    palette = np.array(sns.color_palette('hls', 10))
    ax = plt.subplot(aspect='equal')
    # sc = ax.scatter(feat[:, 0], feat[:, 1], lw=0, s=40, c=palette[label.astype(np.int)])
    for i in range(10):
        plt.plot(feat[label == i, 0], feat[label == i, 1], '.', c=palette[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    ax.axis('tight')
    for i in range(10):
        xtext, ytext = np.median(feat[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])

    plt.draw()
    plt.ioff()
    plt.savefig(output_dir+'/scatter_{}_{}.png'.format(method, epoch), bbox_inches='tight')
