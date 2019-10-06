#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Monday, September 30th 2019, 11:40:06 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sun Oct 06 2019
###

import os
import glob
import random
from shutil import copyfile

import torch
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torchvision import transforms

from server.utils.datasets.img_folder_loader import ImageFolderLoader
from server.utils.load import is_image_file
from server.utils.load import default_loader


SEED = 489
random.seed(489)

class ImageBucket(Dataset):
    def __init__(self, label=0, split=0.8, img_dir='', download_dir='', download_raw=False, output_dir=''):
        super(Dataset, self).__init__()
        
        if output_dir=='':      # If training
            self.LABEL = label
            self.SPLIT= split
            self.IMG_DIR = img_dir
            self.DOWNLOAD_DIR = os.path.join(download_dir, label) 
            self.PROCESSED_DIR = os.path.join(self.DOWNLOAD_DIR, 'processed')
            
            self.transform = transforms.Compose([
                            transforms.Resize((28, 28), interpolation=3),
                            transforms.ToTensor()])
            self.train, self.test = self._load_imgs(download_raw)
        else:
            self.load_dataset(output_dir)

    def __repr__(self):
        return '<ImageBucket {} train: {} test: {} split: {}> \ntransform: {} \ndownload_dir: {} \n' \
                .format(self.LABEL, len(self.train), len(self.test), self.SPLIT, self.transform,
                self.DOWNLOAD_DIR)

    def __len__(self):
        return len(ConcatDataset((self.train, self.test)))

        
    def _load_imgs(self, download_raw):
        PROCESSED_DIR = os.path.join(self.DOWNLOAD_DIR, 'processed')
        TRAIN_PATH = os.path.join(PROCESSED_DIR, 'training.pt')
        TEST_PATH = os.path.join(PROCESSED_DIR, 'test.pt')
        
        # Check if '.pt' data files already exist and load
        if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
            train_data = torch.load(TRAIN_PATH, map_location=lambda storage, loc: storage)
            test_data = torch.load(TEST_PATH, map_location=lambda storage, loc: storage)
            return train_data, test_data
            
            
        label = 0 # every img bucket only has one label
        PROCESSED_DIR = self._mkdirs(PROCESSED_DIR)
        if download_raw:
            TRAIN_DIR = self._mkdirs(os.path.join(self.DOWNLOAD_DIR, 'raw', 'train'))
            TEST_DIR = self._mkdirs(os.path.join(self.DOWNLOAD_DIR, 'raw', 'test'))
            
        fnames = glob.glob(self.IMG_DIR+'/*')
        fnames.sort()    
        random.shuffle(fnames)
        split = int(self.SPLIT * len(fnames))
        train_fnames = fnames[:split]
        test_fnames = fnames[split:]

        print('Processing train images ...')
        train_data = []
        for fp in train_fnames:
            if download_raw:
                fname = os.path.basename(os.path.normpath(fp))
                train_fp = os.path.join(TRAIN_DIR, fname)
                if os.path.exists(train_fp): continue 
                print('Copying ', fp, ' to ', train_fp)
                copyfile(fp, train_fp)    
            if is_image_file(fp):
                img = default_loader(fp)
                train_data.append(self.transform(img))
                            
        print('Processing test images ...')
        test_data = []
        for fp in test_fnames:
            if download_raw:
                fname = os.path.basename(os.path.normpath(fp))
                test_fp = os.path.join(TEST_DIR, fname)
                if os.path.exists(test_fp): continue
                print('Copying ', fp, ' to ', test_fp)
                copyfile(fp, test_fp)
            if is_image_file(fp):
                img = default_loader(fp)
                test_data.append(self.transform(img))
        
        
        train_data = torch.cat(train_data, dim=0).view(-1, 1, 28, 28)  
        test_data = torch.cat(test_data, dim=0).view(-1, 1, 28, 28)  
        train_labels = torch.IntTensor([label for _ in range(len(train_data))])
        test_labels = torch.IntTensor([label for _ in range(len(test_data))])
        train_data = TensorDataset(train_data, train_labels)
        test_data = TensorDataset(test_data, test_labels)
            
        print('Saving datasets to', PROCESSED_DIR)
        torch.save(train_data, TRAIN_PATH)
        torch.save(test_data, TEST_PATH)

        return train_data, test_data


    def _mkdirs(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name
        
    def save_dataset(self, output_dir):
        save_path = os.path.join(output_dir, 'img_bucket.pt')
        torch.save({
            'label': self.LABEL,
            'split': self.SPLIT,
            'transform': self.transform,
            'train': self.train,
            'test': self.test,
            'download_dir': self.DOWNLOAD_DIR
            },
            save_path
            )
    
    def load_dataset(self, output_dir):
        path = os.path.join(output_dir, 'img_bucket.pt')
        dataset = torch.load(path, map_location=lambda storage, loc: storage)
        self.LABEL = dataset['label']
        self.SPLIT = dataset['split']
        self.transform  = dataset['transform']
        self.train = dataset['train']
        self.test = dataset['test']
        self.DOWNLOAD_DIR = dataset['download_dir']

        print('\nLoaded img_bucket.pt from {}\n'.format(output_dir))
