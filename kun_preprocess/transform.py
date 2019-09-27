#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Friday, September 27th 2019, 2:20:58 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Sep 27 2019
###

import numpy as np
import cv2


def dilate(edges, N, iterations): 
    '''Dilate using an NxN '+' sign shape. ary is np.uint8.'''
    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[int((N-1)/2),:] = 1
    dilated_image = cv2.dilate(edges, kernel, iterations=iterations)
    # Image.fromarray(dilated_image*255).show()
    kernel = np.zeros((N+1,N+1), dtype=np.uint8)    # slightly longer kernel for vertical lines
    kernel[:,int((N-1)/2)] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations) 
    return dilated_image


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

