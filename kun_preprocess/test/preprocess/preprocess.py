#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, September 25th 2019, 12:09:26 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Sep 27 2019
###

import glob
import os
import random
import sys
import random
import math
import json
from collections import defaultdict
from io import BytesIO

import cv2
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage.filters import rank_filter


def downscale_image(img, max_dim=2048):
    '''Shrink img until its longest dimension is <= max_dim.
    Returns new_image, scale (where scale <= 1).
    '''
    height, width, depth = img.shape
    if max(height, width) <= max_dim:
        return 1.0, img
    scale = 1.0 * max_dim / max(height, width)
    new_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    
    return scale, new_img

def find_border(contours, ary):
    border = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        if w * h > 0.5 * area:  # Assume border is > 0.5 area
            border= [i, x, y, x + w , y + h ]
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
        border_contour = cv2.boxPoints(r).astype(np.int64) # contour pts must be int64
        
    return border_xy, border_contour

def order_points(pts):
	# init a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = 'float32')
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect
    
def four_point_transform(img, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a 'birds eye view',
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = 'float32')
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped



### Adapted from https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.htmls
def plt_bbox(input_img, num_merge):
    bbox_img = input_img
    img_h, img_w = input_img.shape[0], input_img.shape[1]
    area = img_h*img_w
    for num in range(num_merge):
        thresh = 100 # initial threshold
        edges = cv2.Canny(bbox_img.astype(np.uint8), thresh, thresh*2)
        contours, hierarchy  = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [None]*len(contours)
        bbox = [None]*len(contours)
        bbox_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            bbox[i] = cv2.boundingRect(contours_poly[i])    # x, y, w, h d
            if bbox[i][2] * bbox[i][3] < 0.0001 * area: continue     # skip tiny bboxes
            color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            bbox_img = cv2.rectangle(bbox_img, (int(bbox[i][0]), int(bbox[i][1])), \
            (int(bbox[i][0]+bbox[i][2]), int(bbox[i][1]+bbox[i][3])), color, -1)   # x, y, x+w, x+h
        
        bbox_img = Image.fromarray(bbox_img)
        bbox_img.show()
        bbox_img.convert('RGB').save('7_bbox_{}.png'.format(num))
        bbox_img = np.asarray(bbox_img)
    return bbox, bbox_img
    
def dilate(edges, N, iterations): 
    '''Dilate using an NxN '+' sign shape. ary is np.uint8.'''
    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[int((N-1)/2),:] = 1
    dilated_image = cv2.dilate(edges, kernel, iterations=iterations)
    # Image.fromarray(dilated_image*255).show()
    kernel = np.zeros((N,N), dtype=np.uint8)    # slightly longer kernel for vertical lines
    kernel[:,int((N-1)/2)] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations) 
    return dilated_image


def process_image(path, out_path):
    orig_im = cv2.imread(path)
    scale, img = downscale_image(orig_im)
    
    img =  cv2.GaussianBlur(img, (3,3), 0)   #Remove high freq noise
    
    gaussian_img = Image.fromarray(img)
    gaussian_img.show()
    gaussian_img.save('1_gaussian_blur.png', 'PNG')

    # Edge Detection
    thresh = 100
    edges = cv2.Canny(img, thresh, thresh*2)

    canny_img = Image.fromarray(edges)
    canny_img.show()
    canny_img.convert('RGB').save('2_canny.png', 'PNG')

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_xy, border_contour = find_border_contour(contours, edges)

    # draw border_contour on img
    border_img = cv2.drawContours(img, [border_contour], -1, (0, 255, 0), 3)   

    border_img = Image.fromarray(border_img)
    draw = ImageDraw.Draw(border_img)
    draw.rectangle(border_xy, outline='red')
    border_img.show()
    border_img.save('3_border_contour.png', 'PNG')
    
    # apply the four point transform to obtain a top-down
    # view of the original image
    img = four_point_transform(np.asarray(img), border_contour.reshape(4, 2))
                                
    deskewed_img = Image.fromarray(img)
    deskewed_img.show() 
    deskewed_img.convert('RGB').save('4_deskewed.png', 'PNG')

    
    edges = cv2.Canny(img, thresh, thresh*2)
    # Remove ~1px borders using a rank filter.
    maxed_rows = rank_filter(edges, -4, size=(1, 20))
    maxed_cols = rank_filter(edges, -4, size=(20, 1))
    edges = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
    
    # import matplotlib.pyplot as plt
    # plt.hist(maxed_rows)
    # plt.show()

    filter_text = Image.fromarray(edges)
    filter_text.show() 
    filter_text.convert('RGB').save('5_filter_text.png', 'PNG')

    dilated = dilate(edges, N=2, iterations=5)  # super sensitive!!

    dilated_img = Image.fromarray(dilated)
    dilated_img.show() 
    dilated_img.convert('RGB').save('6_dilated.png', 'PNG')

    bbox, bbox_img = plt_bbox(dilated, num_merge=2)
    # bbox_img = Image.fromarray(bbox_img)
    # bbox_img.convert('RGB').save('7_bbox.png', 'PNG')
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #remove contour outline
    
    # img = cv2.adaptiveThreshold(img, 
    #                             maxValue=255, 
    #                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                             thresholdType=cv2.THRESH_BINARY, 
    #                             blockSize=115, 
    #                             C=10)   # offset  
    # 
    # Binarising seems to worsen quality... 
    from skimage.filters import threshold_local   
    T = threshold_local(img, 11, offset = 10, method = 'gaussian')
    binarise_img = (img > T).astype('uint8') * 255
    binarise_img = Image.fromarray(binarise_img).convert('RGB')
    binarise_img.show()
    binarise_img.save('8_binarise.png', 'PNG')


    img = Image.fromarray(img).convert('RGB')   # plot bbox

    # Upscaling
    a, b = img.size
    img = img.resize((int(a / scale), int(b / scale)), resample=Image.BICUBIC)  # upscale
    bbox = [[int(x / scale) for x in box] for box in bbox]
    img.save('8_final.png', 'PNG')
    # Clearing folder
    # import shutil
    # shutil.rmtree('cropped')
    # os.mkdir('cropped') 
    # for idx, box in enumerate(bbox):    # Crop the original img
    #     crop = img.crop([box[0], box[1], box[0]+box[2], box[1]+box[3]])
    #     print('Saving cropped/{}.png'.format(idx))
    #     crop.save('cropped/{}.png'.format(idx))

    
    bbox_draw = ImageDraw.Draw(img)
    for box in bbox:
        bbox_draw.rectangle([box[0], box[1], box[0]+box[2], box[1]+box[3]], outline='red', width=5)
    img.show()
    img.save('8_final_bbox.png', 'PNG')

    

    
      
if __name__ == '__main__':
    if len(sys.argv) == 2 and '*' in sys.argv[1]:
        files = glob.glob(sys.argv[1])
        random.shuffle(files)
    else:
        files = sys.argv[1:]

    for path in files:
        out_path = path.replace('.'+path[-3:], '.crop.png')
        print(out_path) 
        if os.path.exists(out_path): 
            continue
        print('Processing') 
        process_image(path, out_path)     

        ## To read error o/p
        # try:
        #     print('Processing')
        #     process_image(path, out_path)
        # except Exception as e:
        #     print ('%s %s Error' % (path, e))