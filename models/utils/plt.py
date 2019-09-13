#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, August 22nd 2019, 9:25:01 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Sep 13 2019
###

import numpy as np
import torch

import sklearn.metrics

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set(font_scale=2)

SEED = 489

def plt_scatter(feat=[], labels=[], colors=[], output_dir='.', plt_name='', pltshow=False):
    print('Plotting {}\n'.format(plt_name))
    labels_list = np.unique(labels[labels!=-1])     # -1 is noise 
    palette = sns.color_palette('hls', labels_list.max()+1)
    feat_colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    ax = plt.subplot()
    ax.tick_params(axis='both', labelsize=10)
    
    if len(colors) == 0:
        plt.scatter(*feat.T, c=feat_colors, s=8, linewidths=1)
    else:
        plt.scatter(*feat[0].T, c=feat_colors, s=8, linewidths=1)
        for i, f in enumerate(feat[1:]):
            plt.scatter(*f.T, c=colors[i], s=8, linewidths=1)
        feat = feat[0]

    for label in labels_list:   
        xtext, ytext = np.median(feat[labels == label, :], axis=0)
        txt = ax.text(xtext, ytext, str(label), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground='w'), PathEffects.Normal()])
    
    plt.savefig(output_dir+'/'+plt_name, bbox_inches='tight')
    if pltshow:
        plt.show()
    plt.close()
    return plt.imread(output_dir+'/'+plt_name)

def plt_scatter_3D(feat=[], labels=[], colors=[], output_dir='.', plt_name='', pltshow=False):
    print('Plotting {}\n'.format(plt_name))
    labels_list = np.unique(labels[labels!=-1])     # -1 is noise 
    palette = sns.color_palette('hls', labels_list.max()+1)
    feat_colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(*feat.T, c=feat_colors, s=8, linewidths=1)

    if len(colors) == 0:
        ax.scatter(*feat.T, c=feat_colors, s=8, linewidths=1)
    else:
        ax.scatter(*feat[0].T, c=feat_colors, s=8, linewidths=1)
        for i, f in enumerate(feat[1:]):
            ax.scatter(*f.T, c=colors[i], s=8, linewidths=1)
        feat = feat[0]

    for label in labels_list:   
        xtext, ytext, ztext = np.median(feat[labels == label, :], axis=0)
        txt = ax.text(xtext, ytext, ztext, str(label), fontsize=18)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground='w'), PathEffects.Normal()])
    
    plt.savefig(output_dir+'/'+plt_name, bbox_inches='tight')
    if pltshow:
        plt.show()
    plt.close()
    return plt.imread(output_dir+'/'+plt_name)      
        
def plt_confusion_matrix(y_pred, y_target, output_dir, pltshow=False):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_target, y_pred)

    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', annot_kws={'size': 20})
    # plt.title('Confusion matrix', fontsize=10)
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Clustering label', fontsize=20)
    plt.savefig(output_dir+'/confusion_matrix.png', bbox_inches='tight')

    if pltshow:
        plt.show()

    return plt.imread(output_dir+'/confusion_matrix.png'), 'confusion_matrix.png'


def plt_clusters(output_dir, data, algorithm, args, kwds):
    fig = plt.figure()
    # start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    # end_time = time.time()
    ax = plt.subplot()
    # ax.axis('tight')
    ax.tick_params(axis='both', labelsize=10)
    # labels_list = np.unique(labels)
    # palette = np.array(sns.color_palette('hls', len(labels_list)))
    palette = sns.color_palette('hls')
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, s=8, linewidths=1)
    # frame = plt.gca()
    # frame.axes.get_xaxis().set_visible(False)
    # frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=14)
    #plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    plt.savefig(output_dir)
    print ( '\n  saved image ', output_dir)
    plt.close(fig)
    return plt.imread(output_dir)


    