#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Monday, September 30th 2019, 11:40:06 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Tue Oct 01 2019
###

import os
import glob
import random
import matplotlib as plt
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from shutil import copyfile

# from server.utils.datasets.img_folder_loader import ImageFolderLoader
# from server.utils.datasets.img_folder_loader import is_image_file
from img_folder_loader import ImageFolderLoader


SEED = 489
# torch.manual_seed(SEED)    # reproducible
random.seed(489)

class ImageBucket(Dataset):
    def __init__(self, label=0, split=0.8, img_dir='', download_raw=True, download_dir='', output_dir=''):
        super(Dataset, self).__init__()
        
        if output_dir=='':      # If training
            self.LABEL = label
            self.SPLIT= split
            self.IMG_DIR = img_dir
            self.DOWNLOAD_DIR = os.path.join(download_dir, label) 
            self.transforms = transforms.Compose([
                            transforms.Resize((28, 28), interpolation=3),
                            transforms.ToTensor()]) 
            self.train, self.test = self._load_imgs(download_raw)
        else:
            self.load_dataset(output_dir)
        
        

    def _load_imgs(self, download_raw):
        self.PROCESSED_DIR = self._mkdirs(os.path.join(self.DOWNLOAD_DIR, 'processed'))
        if download_raw:
            self.TRAIN_DIR = self._mkdirs(os.path.join(self.DOWNLOAD_DIR, 'raw', 'train'))
            self.TEST_DIR = self._mkdirs(os.path.join(self.DOWNLOAD_DIR, 'raw', 'test'))
            
        fnames = glob.glob(self.IMG_DIR+'/*')
        fnames.sort()    
        random.shuffle(fnames)

        split = int(self.SPLIT * len(fnames))
        train_fnames = fnames[:split]
        test_fnames = fnames[split:]

        print('Processing train images ...')
        
        for fp in train_fnames:
            if download_raw:
                fname = os.path.basename(os.path.normpath(fp))
                train_fp = os.path.join(self.TRAIN_DIR, fname)
                if os.path.exists(train_fp): continue 
                print('Copying ', fp, ' to ', train_fp)
                copyfile(fp, train_fp)

        print('Processing test images ...')
        for fp in test_fnames:
            if download_raw:
                fname = os.path.basename(os.path.normpath(fp))
                test_fp = os.path.join(self.TEST_DIR, fname)
                if os.path.exists(test_fp): continue
                print('Copying ', fp, ' to ', test_fp)
                copyfile(fp, test_fp)
            
        if download_raw:
            train_data = ImageFolderLoader(self.TRAIN_DIR, label=self.LABEL, transform = self.transforms )
            test_data = ImageFolderLoader(self.TEST_DIR, label=self.LABEL, transform = self.transforms )
                # img = cv2.imread(fp)

        return train_data, test_data
    
    
    def _mkdirs(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name
        
    def save_dataset(self, output_dir):
        save_path = os.path.join(output_dir, 'img_bucket.pt')
        torch.save({
            'label': self.LABEL,
            'n_noise_clusters': self.N_NOISE_CLUSTERS,
            'split': self.SPLIT,
            'train': self.train,
            'test': self.test
            },
            save_path
            )
    
    def load_dataset(self, output_dir):
        path = os.path.join(output_dir, 'img_bucket.pt')
        dataset = torch.load(path, map_location=lambda storage, loc: storage)
        self.LABEL = dataset['label']
        self.N_NOISE_CLUSTERS = dataset['n_noise_clusters']
        self.SPLIT = dataset['split']
        self.train = dataset['train']
        self.test = dataset['test']

        print('\nLoaded img_bucket.pt from {}\n'.format(output_dir.split('/')[1]))
