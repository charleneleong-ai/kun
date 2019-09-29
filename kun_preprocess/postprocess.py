#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Saturday, September 28th 2019, 5:13:10 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sat Sep 28 2019
###

import os
import glob
import argparse

import cv2
from PIL import Image, ImageDraw
import numpy as np

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('directory', help='Directory where image files are.')
    # args = parser.parse_args()
    # parser = argparse.ArgumentParser()

    files = glob.glob('char_output/*/*/*.png')
    
    outdir = 'char_output_processed'
    if not os.path.exists(outdir):
        os.mkdir(outdir)


    # for fp in files:
    fp = files[0]
    fn = os.path.basename(os.path.normpath(fp))
    
    img = cv2.imread(fp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    edges = cv2.Canny(img, lowThresh, high_thresh)
    Image.fromarray(edges).show()