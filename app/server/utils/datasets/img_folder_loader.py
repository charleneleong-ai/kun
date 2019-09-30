#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Monday, September 30th 2019, 7:48:30 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Tue Oct 01 2019
###

import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolderLoader(Dataset):
    """Assumes download_dir = root/label/raw/train/*.png"""
    def __init__(self, download_dir, label='', 
                 transform=None, 
                 loader=default_loader):
        super(Dataset, self).__init__()

        self.DOWNLOAD_DIR = download_dir    
        self.LABEL = label
        self.transform = transform
        self.loader = loader
        self.imgs = self._mkdataset()

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(os.path.join(self.DOWNLOAD_DIR, path))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def _mkdataset(self):
        images = []
        for fp in os.listdir(self.DOWNLOAD_DIR):
            if is_image_file(fp):
                images.append((fp, self.LABEL))
        return images

    def _mkdirs(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name
    
    