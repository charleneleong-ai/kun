#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, September 12th 2019, 9:53:52 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Sep 16 2019
###

import numpy as np
SEED = 489
np.random.seed(SEED)

class SOM(object):
    def __init__(self, data, dims, n_iter, lr):
        self.DIMS = np.array(dims)
        self.N_ITER = n_iter
        self.LR = lr

        # establish size variables based on data
        self.data = data
        self.N = data.shape[0]  # num of samples
        self.M = data.shape[1]  # num of dims

        # weight matrix (i.e. the SOM) needs to be a m-dim vector for each neuron in the SOM
        # setup random weights between 0 and 1
        self.net = np.random.random((dims[0], dims[1], self.M))
        
        # initial neighbourhood radius
        self.RADIUS = max(dims[0], dims[1]) / 2
        # radius decay parameter
        self.TIME_CONSTANT = n_iter / np.log(self.RADIUS)

    def __repr__(self):
        return '<SOM: {} N_ITER: {} LR: {}>'.format(self.DIMS, self.N_ITER, self.LR)

    def find_bmu(self, t, net, m):
        #  Find the best matching unit for a given vector, t, in the SOM
        #  Returns: a (bmu, bmu_idx) tuple where bmu is the high-dimensional BMU
        #                 and bmu_idx is the index of this vector in the SOM
        
        bmu_idx = np.array([0, 0])
        # set the initial minimum distance to a huge number
        min_dist = np.iinfo(np.int).max
        # calculate the high-dimensional distance between each neuron and the input
        for x in range(net.shape[0]):
            for y in range(net.shape[1]):
                w = net[x, y, :].reshape(m, 1)
                # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
                dist = np.sum((w - t) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    bmu_idx = np.array([x, y])
        # get vector corresponding to bmu_idx
        bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
        # return the (bmu, bmu_idx) tuple
        return (bmu, bmu_idx)

    def train(self):
        for i in range(self.N_ITER):

            if i % 100 == 0: print('Iteration %d' % i)
            
            # select a training example at random
            t = self.data[np.random.randint(0, self.N), :].reshape(np.array([self.M, 1]))
   
            # find its Best Matching Unit
            bmu, bmu_idx = self.find_bmu(t, self.net, self.M)
            
            # decay the SOM parameters
            radius = self.RADIUS * np.exp(-i / self.TIME_CONSTANT)
            lr = self.LR * np.exp(-i / self.N_ITER)
            
            # update weight vector to move closer to input
            # and move its neighbours in 2-D vector space closer
            # by a factor proportional to their 2-D distance from the BMU
            for x in range(self.net.shape[0]):
                for y in range(self.net.shape[1]):
                    w = self.net[x, y, :].reshape(self.M, 1)
                    # get the 2-D distance (again, not the actual Euclidean distance)
                    w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                    # w_dist = np.sqrt(w_dist)
                    
                    if w_dist <= radius**2:
                        # calculate the degree of influence (based on the 2-D distance)
                        influence = np.exp(-w_dist / (2* (radius**2)))
                        
                        # new w = old w + (learning rate * influence * delta)
                        # where delta = input vector (t) - old w
                        new_w = w + (lr * influence * (t - w))
                        self.net[x, y, :] = new_w.reshape(1, self.M)
        
        return self.net
    
    def get_net_nn(self):
        centroids = np.array([net[x-1, y-1, :] for x in range(net.shape[0]) for y in range(net.shape[1])])
        
