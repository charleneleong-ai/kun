#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, August 28th 2019, 3:25:31 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Thu Sep 05 2019
###

import numpy as np


class EarlyStopping(object):
    def __init__(self, mode='min', tol=0, patience=10, percentage=False):
        self.mode = mode
        self.tol = tol
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, tol, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1
            print('Loss did not improve from {:.6f} for {} epochs\n'.format(metrics, self.num_bad_epochs))

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, tol, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - tol
            if mode == 'max':
                self.is_better = lambda a, best: a > best + tol
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * tol / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * tol / 100)