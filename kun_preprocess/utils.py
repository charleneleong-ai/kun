#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Friday, September 27th 2019, 1:36:19 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Sep 27 2019
###

import cv2
import numpy as np
import random
from PIL import Image

def resize(image, width=None, height=None):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def downscale(img, max_dim=2048):
    '''Shrink img until its longest dimension is <= max_dim.
    Returns new_image, scale (where scale <= 1).
    '''
    height, width, depth = img.shape
    if max(height, width) <= max_dim:
        return 1.0, img
    scale = 1.0 * max_dim / max(height, width)
    new_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    
    return scale, new_img


### Adapted from https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.htmls
def plt_bbox(input_img, num_merge, pltshow=False):
    bbox_img = input_img
    img_h, img_w = input_img.shape[0], input_img.shape[1]
    area = img_h*img_w
    for _ in range(num_merge):
        thresh = 100 # initial threshold
        edges = cv2.Canny(bbox_img.astype(np.uint8), thresh, thresh*2)
        contours, hierarchy  = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [None]*len(contours)
        bbox = [None]*len(contours)
        bbox_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            bbox[i] = cv2.boundingRect(contours_poly[i])    # x, y, w, h d
            if bbox[i][2] * bbox[i][3] < 0.001 * area: continue     # skip tiny bboxes
            color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            bbox_img = cv2.rectangle(bbox_img, (int(bbox[i][0]), int(bbox[i][1])), \
            (int(bbox[i][0]+bbox[i][2]), int(bbox[i][1]+bbox[i][3])), color, -1)   # x, y, x+w, x+h
        
        if pltshow:
            Image.fromarray(bbox_img).show()
        
    return bbox, bbox_img

def save_image(img, path, pltshow):
    if type(img) == np.ndarray:
        img = Image.fromarray(img)
    if pltshow:
        img.show()
    img.convert('RGB').save(path)