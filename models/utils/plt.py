#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, August 22nd 2019, 9:25:01 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Aug 30 2019
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
    print('Plotting {}_{}.png \n'.format(method, epoch))
    
    if feat.shape[1] > 2:            # Reduce to 2 dim
        if feat.shape[0] > 5000:     # Plot only first 5000 pts   
            feat = feat[:5000, :]
            labels = labels[:5000]

        if method == 'pca':
            pca = PCA(n_components=2, random_state=SEED)
            feat = pca.fit_transform(feat)
        elif method == 'tsne':
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=SEED)
            feat = tsne.fit_transform(feat)

    labels_list = np.unique(labels)
    
    plt.ion()
    plt.clf()
    palette = np.array(sns.color_palette('hls', len(labels_list)))
    ax = plt.subplot(aspect='equal')
    
    for i, label in enumerate(labels_list):
        plt.plot(feat[labels == label, 0], feat[labels == label, 1], '.', c=palette[i])

        ax.axis('tight')
        ax.tick_params(axis='both', labelsize=10)
        xtext, ytext = np.median(feat[labels == label, :], axis=0)
        txt = ax.text(xtext, ytext, str(label), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])

    plt.draw()
    plt.ioff()
    plt.savefig(output_dir+'/{}_{}.png'.format(method, epoch), bbox_inches='tight')

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



def plt_clusters(outfile, data, algorithm, args, kwds):
    fig = plt.figure()
    # start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    # end_time = time.time()
    # palette = sns.color_palette('deep', np.unique(labels).max() + 1)

    labels_list = np.unique(labels)
    palette = np.array(sns.color_palette('hls', len(labels_list)))
    
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, alpha=0.5, s=8, linewidths=1)
    # frame = plt.gca()
    # frame.axes.get_xaxis().set_visible(False)
    # frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=14)
    #plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    plt.savefig(outfile)
    print ( '\n  saved image ',outfile )
    plt.close(fig)
    return    


    