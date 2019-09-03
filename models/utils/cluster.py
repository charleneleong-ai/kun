#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, August 28th 2019, 3:28:39 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Aug 30 2019
###

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

SEED = 489

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

def find_nn(centroids, data):
    closest, _ = pairwise_distances_argmin_min(centroids, data)
    return closest

def find_k_nn(centroids, data, k=1, distance_norm=2):
    """
    Arguments:
    ----------
        centroids: (M, d) ndarray
            M - number of clusters
            d - number of data dimensions
        data: (N, d) ndarray
            N - number of data points
        k: int (default 1)
            nearest neighbour to get
        distance_norm: int (default 2)
            1: Hamming distance (x+y)
            2: Euclidean distance (sqrt(x^2 + y^2))
            np.inf: maximum distance in any dimension (max((x,y)))

    Returns:
    -------
        indices: (M,) ndarray
        values: (M, d) ndarray
    """

    kdtree = cKDTree(data)
    distances, indices = kdtree.query(centroids, k, p=distance_norm)
    if k > 1:
        indices = indices[:,-1]
    values = data[indices]
    return indices, values

    # indices, values = find_k_closest(centroids, feat)
    # print(indices)

def find_n_clusters_bic(feat, output_dir=""):
    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='full', random_state=SEED).fit(feat) for n in n_components]
    bic = [m.bic(feat) for m in models]
    aic = [m.aic(feat) for m in models]

    ymin = min(bic)     # Finding min pt
    xmin = bic.index(ymin)
    k = n_components[xmin]
    print(k, ' components')

    if plt:
        fig = plt.figure()
        plt.plot(n_components, bic, label='BIC')
        plt.plot(n_components, aic,  label='AIC')
        plt.legend(loc='best')
        plt.xlabel('n_components')
        plt.savefig(output_dir+'/optimal_k_bic.png', bbox_inches='tight')
        plt.close(fig)
    
    return k