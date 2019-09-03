#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, August 28th 2019, 3:28:39 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Aug 30 2019
###

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

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

def find_closest(centroids, data):
    closest, _ = pairwise_distances_argmin_min(centroids, data)


def find_k_closest(centroids, data, k=1, distance_norm=2):
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
