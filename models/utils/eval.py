#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, August 28th 2019, 3:28:39 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Aug 30 2019
###

import numpy as np
from scipy.optimize import linear_sum_assignment

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