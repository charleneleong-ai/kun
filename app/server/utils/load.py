#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Monday, September 23rd 2019, 1:42:53 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Sep 23 2019
###

import numpy as np
import codecs, json

def np_json(np_array, file_path):
    return json.dump(np_array.tolist(), codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    
def json_np(file_path):
    return np.array(json.loads((codecs.open(file_path, 'r', encoding='utf-8').read())))


    