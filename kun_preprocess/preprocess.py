#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, September 25th 2019, 12:09:26 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sun Sep 29 2019
###

import glob
import os
import sys

import cv2
from PIL import Image, ImageDraw
import numpy as np

from transform import downscale, resize, dilate, four_point_transform, auto_canny
from utils import find_border_contour, save_img

def process_image(path, out_path):
    orig_im = cv2.imread(path)
    if type(orig_im) == None:
        print('Error loading image')
        raise ValueError('Error loading image')
    
    scale, img = downscale(orig_im)
    
    # img =  cv2.GaussianBlur(img, (3,3), 0)   #Remove high freq noise
    # save_img(img, path='1_gaussian_blur.png', pltshow=True)

    img = cv2.bilateralFilter(img, 7, 50, 50)
    # save_img(img, path='1_bilateral_blur.png', pltshow=True)

    # Edge Detection
    edges = auto_canny(img)
    # save_img(edges, path='2_canny_bilateral.png', pltshow=True)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_xy, border_contour = find_border_contour(contours, edges)

    # draw border_contour on img
    # border_img = cv2.drawContours(img, [border_contour], -1, (0, 255, 0), 3)   
    # border_img = Image.fromarray(border_img)
    # draw = ImageDraw.Draw(border_img)
    # draw.rectangle(border_xy, outline='red')
    # save_img(border_img, path='3_border_contour.png', pltshow=True)


    # apply the four point transform to obtain a top-down
    # view of the original image
    img = four_point_transform(np.asarray(img), border_contour.reshape(4, 2))
    # save_img(img, path='4_deskewed.png', pltshow=True)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #remove contour outline
    # img = Image.fromarray(img).convert('RGB')  
    a, b = img.shape
    img = resize(np.asarray(img), width=int(a / scale), height=int(b / scale))
    #img = img.resize((int(a / scale), int(b / scale)), resample=Image.BICUBIC)  # upscale
    img = Image.fromarray(img)
    # img.show()
    img.save(out_path, 'PNG')


    
if __name__ == '__main__':
    if len(sys.argv) == 2 and '*' in sys.argv[1]:
        files = glob.glob(sys.argv[1])
    else:
        files = sys.argv[1:]

    outputdir = 'NZCGMJ_processed'
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    for path in files:
        fname = os.path.basename(os.path.normpath(path))
        fname, ext = os.path.splitext(fname)
        if ext != '.tif': 
            print('Not a image file')
            continue
        fname = fname+ '.png'    # Change to png
        out_path = os.path.join('NZCGMJ_processed', fname)
        if os.path.exists(out_path): 
            continue
        
        ## To read error o/p
        try:
            print('Processing ' + path)
            process_image(path, out_path)
        except Exception as e:
            with open(os.path.join('NZCGMJ_processed', 'error_files.txt'), 'a') as f: 
                f.write('{} {}  Error \n'.format(fname, e))

            print ('{} {} Error'.format(fname, e))