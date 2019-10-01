#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Monday, September 23rd 2019, 1:42:53 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Tue Oct 01 2019
###

import numpy as np
import codecs, json
from PIL import Image

def np_json(np_array, file_path):
    return json.dump(np_array.tolist(), codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    
def json_np(file_path):
    return np.array(json.loads((codecs.open(file_path, 'r', encoding='utf-8').read())))


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('L')    # (8-bit pixels, black and white) match MNIST
