#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Friday, September 27th 2019, 2:20:58 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Sep 30 2019
###

import numpy as np
from PIL import Image
import cv2

# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged


def dilate(edges, N, iterations):
    '''Dilate using an NxN '+' sign shape. ary is np.uint8.'''
    kernel = np.zeros((N, N), dtype=np.uint8)
    kernel[int((N-1)/2), :] = 1
    dilated_image = cv2.dilate(edges, kernel, iterations=iterations)
    # Image.fromarray(dilated_image*255).show()
    kernel = np.zeros((N, N), dtype=np.uint8)
    kernel[:, int((N-1)/2)] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    return dilated_image


def center_img(bbox=[], img_crop=[]):
	b_x, b_y, b_w, b_h = bbox
	x1, y1, w, h = img_crop
	b_cx = x1+b_x+(b_w/2)
	b_cy = y1+b_y+(b_h/2)
	img_cx = x1+(w/2)
	img_cy = y1+(h/2)
	x_trans = img_cx-b_cx  # orig_c_x - c_x
	y_trans = img_cy-b_cy     #orig_c_y - c_y
	return translate_bbox(x_trans, y_trans, img_crop)


def translate_bbox(x_trans, y_trans, bbox=[]):  # x, y, w, h
    new_bbox=[]
    if x_trans > 0:
        x = bbox[0]+ x_trans
        new_bbox.append(x)    # moving right
    else:   
        x = bbox[0]-x_trans
        new_bbox.append(x)    # moving left

    if y_trans > 0:
        y = bbox[1]-y_trans     # moving up
        new_bbox.append(y)
    else:
        y = bbox[1]+y_trans
        new_bbox.append(y)      # moving down
    
    new_bbox.append(bbox[2])
    new_bbox.append(bbox[3])

    return new_bbox


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

def resize(img, width=None, height=None):

    (h, w) = img.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def downscale(img, max_dim=2048):
    # Shrink img until its longest dimension is <= max_dim.
    # Returns new_image, scale (where scale <= 1).
    height, width, depth = img.shape
    if max(height, width) <= max_dim:
        return 1.0, img
    scale = 1.0 * max_dim / max(height, width)
    new_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    
    return scale, new_img

def square_img(img, min_size=28, fill_color=(255, 255, 255, 255)): # fill white
    if type(img) == np.ndarray:
        img = Image.fromarray(img)
    h, w = img.size
    size = max(min_size, h, w)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(img, (int((size - h) / 2), int((size - w) / 2)))
    return new_im

def make_square(img_crop, min_size=28):
	x1, y1, w, h = img_crop

	if h < min_size:
		h = min_size
	if w<min_size:
		w = min_size
		
	if h-w > 0: # taller
		x1 = x1
		y1 = y1+0.5*(h-w)
		h = w
	else: # wider
		x1 = x1+0.5*(h-w)
		y1 = y1
		w = h

	return [x1, y1, w, h]