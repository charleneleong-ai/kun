#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, September 25th 2019, 12:09:26 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Sep 27 2019
###

import glob
import os
import sys

import cv2
from PIL import Image, ImageDraw
import numpy as np
import random
from scipy.ndimage.filters import rank_filter

from utils import downscale, resize, save_image
from transform import find_border_contour, dilate, four_point_transform


def process_image(path, out_path):
    orig_im = cv2.imread(path)
    scale, img = downscale(orig_im)
    
    img =  cv2.GaussianBlur(img, (3,3), 0)   #Remove high freq noise
    # save_image(img, path='1_gaussian_blur.png', pltshow=True)

    # Edge Detection
    thresh = 100
    edges = cv2.Canny(img, thresh, thresh*2)
    # save_image(edges, path='2_canny.png', pltshow=True)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_xy, border_contour = find_border_contour(contours, edges)

    # draw border_contour on img
    border_img = cv2.drawContours(img, [border_contour], -1, (0, 255, 0), 3)   
    border_img = Image.fromarray(border_img)
    draw = ImageDraw.Draw(border_img)
    draw.rectangle(border_xy, outline='red')
    # save_image(border_img, path='3_border_contour.png', pltshow=True)


    # apply the four point transform to obtain a top-down
    # view of the original image
    img = four_point_transform(np.asarray(img), border_contour.reshape(4, 2))
    # save_image(img, path='4_deskewed.png', pltshow=True)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #remove contour outline
    img = Image.fromarray(img).convert('RGB')  
    a, b = img.size
    img = resize(np.asarray(img), width=int(a / scale), height=int(b / scale))
    #img = img.resize((int(a / scale), int(b / scale)), resample=Image.BICUBIC)  # upscale
    img = Image.fromarray(img)
    # img.show()
    print('Saving ' + out_path)
    img.save(out_path, 'PNG')


    
if __name__ == '__main__':
    if len(sys.argv) == 2 and '*' in sys.argv[1]:
        files = glob.glob(sys.argv[1])
        random.shuffle(files)
    else:
        files = sys.argv[1:]

    for path in files:
        fname = os.path.basename(os.path.normpath(path))
        fname = fname.replace(path[-3:], '.png')    # Change to png
        out_path = os.path.join('NZCGMJ_processed', fname)
        if os.path.exists(out_path): 
            continue
        
        ## To read error o/p
        try:
            print('Processing ' + path)
            process_image(path, out_path)
        except Exception as e:
            with open(os.path.join('NZCGMJ_processed', 'error_files.txt'), 'a') as f: 
                f.write('%s %s Error \n' % (fname, e))

            print ('%s %s Error' % (fname, e))