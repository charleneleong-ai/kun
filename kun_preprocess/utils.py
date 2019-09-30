#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Friday, September 27th 2019, 1:36:19 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Sep 30 2019
###

import cv2
import numpy as np
import random
from PIL import Image
from transform import auto_canny, dilate


# Adapted from https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.htmls
def plt_bbox_img(input_img, num_merge, pltshow=False):
    bbox_img = input_img
    img_h, img_w = input_img.shape[0], input_img.shape[1]
    area = img_h*img_w
    for _ in range(num_merge):
        edges = auto_canny(bbox_img.astype(np.uint8))
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [None]*len(contours)
        bbox = [None]*len(contours)
        bbox_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            bbox[i] = cv2.boundingRect(contours_poly[i])    # x, y, w, h d
            # skip tiny bboxes
            if bbox[i][2] * bbox[i][3] < 0.001 * area: continue
            color = (random.randint(0, 256), random.randint(
                0, 256), random.randint(0, 256))
            bbox_img = cv2.rectangle(bbox_img, (int(bbox[i][0]), int(bbox[i][1])),
            (int(bbox[i][0]+bbox[i][2]), int(bbox[i][1]+bbox[i][3])), color, -1)   # x, y, x+w, x+h

        if pltshow:
            Image.fromarray(bbox_img).show()

    return bbox, bbox_img

def find_contours(img):
    edges = auto_canny(np.asarray(img).astype(np.uint8))
    edges = dilate(edges, N=3, iterations=4)
    # Image.fromarray(edges).show()
    return cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def find_bbox(img):
    contours, hierarchy = find_contours(img)
    # Assume largest contour is img
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = contours[0]
    contours_poly = cv2.approxPolyDP(c, 3, True)     # Approx contour
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(img,[box],0,(0,0,255),2)
    return cv2.boundingRect(contours_poly)    # x, y, w, h


def add_padding_img(img, bbox=[], img_crop=[], padding=20, pltshow=False):
    b_x, b_y, b_w, b_h = bbox
    x1, y1, w, h = img_crop
    x2 = x1+w   # bot_right
    y2 = y1+h
    
    x1_diff = (x1 + b_x) - x1 
    y1_diff = (y1 + b_y) - y1
    x2_diff = x2 - (b_x+b_w)
    y2_diff = y2 - (b_y+b_h)
    
    crop = img.crop([x1, y1, x2, y2])
    while (x1_diff < padding or y1_diff < padding or x2_diff < padding or y2_diff < padding):
        if x1_diff < padding:   # left
            x1 = x1 - padding
        if y1_diff < padding:   # top
            y1 = y1 - padding
        if x2_diff < padding:   # right
            x2 = x2 + padding
        if y2_diff < padding:   # bot
            y2 = y2 + padding

        crop = img.crop([x1, y1, x2, y2])
        if pltshow:
            crop.show()
            
        b_x, b_y, b_w, b_h = find_bbox(crop)
        x1_diff = (x1 + b_x) - x1
        y1_diff = (y1 + b_y) - y1
        x2_diff = x2 - (b_x+b_w)
        y2_diff = y2 - (b_y+b_h)

    return crop, [x1, y1, (x2-x1), (y2-y1)], [b_x, b_y, b_w, b_h]     # x, y, w, h

def add_padding_bbox(bbox=[], img_crop=[], padding=20):
    x1, y1, w, h = img_crop
    b_x, b_y, b_w, b_h = bbox
    
    if b_x > padding:  # Adding padding top left if possible
        b_x = b_x-padding
        b_w = b_w+padding    # compensate
    else:
        b_w = b_x+b_w   
        b_x = 0
        
    if b_y > padding:
        b_y = b_y-padding
        b_h = b_h+padding # compensate
    else:
        b_h = b_y+b_h
        b_y = 0

    if (w-b_w) > padding:
        b_w = b_w+padding
    else:
        b_w = b_w+(w-b_w)
            
    if (h-b_h) > padding:
        b_h = b_h+padding
    else:
        b_h = b_h+(h-b_h)
    
    return [b_x, b_y, b_w, b_h]
    

def find_border(contours, ary):
    border = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x, y, w, h=cv2.boundingRect(c)
        if w * h > 0.5 * area:  # Assume border is > 0.5 area
            border = [i, x, y, x + w, y + h]
    return border

def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))

def find_border_contour(contours, edges):
    border = find_border(contours, edges)

    border_xy = border[1:]  # [i, x1, y1, x2, y2]
    border_contour = contours[border[0]]

    r = cv2.minAreaRect(border_contour)
    degs = r[2]
    if angle_from_right(degs) <= 10.0:
        border_contour = cv2.boxPoints(r).astype(
            np.int64)  # contour pts must be int64

    return border_xy, border_contour

def save_img(img, path, pltshow):
    if type(img) == np.ndarray:
        img = Image.fromarray(img)
    if pltshow:
        img.show()
    img.convert('RGB').save(path)
