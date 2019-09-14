#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Saturday, September 14th 2019, 12:13:46 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sat Sep 14 2019
###
import os
CURRENT_FNAME = __file__.split('.')[0]
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURRENT_DIR, CURRENT_FNAME)

import torch
from model.ae import AutoEncoder

ae = AutoEncoder()
print (ae)