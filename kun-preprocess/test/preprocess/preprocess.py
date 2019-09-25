#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, September 25th 2019, 12:09:26 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Thu Sep 26 2019
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

# def props_for_contours(contours, ary):
#     '''Calculate bounding box & the number of set pixels for each contour.'''
#     c_info = []
#     for c in contours:
#         x,y,w,h = cv2.boundingRect(c)
#         c_im = np.zeros(ary.shape)
#         cv2.drawContours(c_im, [c], 0, 255, -1)
#         c_info.append({
#             'x1': x,
#             'y1': y,
#             'x2': x + w - 1,
#             'y2': y + h - 1,
#             'sum': np.sum(ary * (c_im > 0))/255
#         })
#     return c_info


# def union_crops(crop1, crop2):
#     '''Union two (x1, y1, x2, y2) rects.'''
#     x11, y11, x21, y21 = crop1
#     x12, y12, x22, y22 = crop2
#     return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


# def intersect_crops(crop1, crop2):
#     x11, y11, x21, y21 = crop1
#     x12, y12, x22, y22 = crop2
#     return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)


# def crop_area(crop):
#     x1, y1, x2, y2 = crop
#     return max(0, x2 - x1) * max(0, y2 - y1)





# def find_components(edges, max_components=80):
#     '''Dilate the image until there are just a few connected components.
#     Returns contours for these components.'''
#     # Perform increasingly aggressive dilation until there are just a few
#     # connected components.
#     count = max_components+1
#     n = 1

#     while count > max_components:
#         n += 1
#         dilated_image = dilate(edges, N=3, iterations=n)
#         if (count % 5 == 0):
#             print(count)
#             img = Image.fromarray(255 * dilated_image)
#             img.show()
#             img.convert('RGB').save('dilated/{}_{}.png'.format(n, count), 'PNG')
#         # Find contours only accepts uint8, need to normalise dilated_image output 
#         contours, hierarchy = cv2.findContours(dilated_image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         count = len(contours)
        
#         print(n, count)
    
#     img = Image.fromarray(255 * dilated_image)
#     img.show()
#     img.convert('RGB').save('dilated/{}_{}.png'.format(n, count), 'PNG')

#     draw = ImageDraw.Draw(img)
#     c_info = props_for_contours(contours, edges)
    
#     for c in c_info:
#        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
#        draw.rectangle(this_crop, outline='blue')
#     # draw.rectangle(crop, outline='red')
#     print(count)
#     img.show()
#     return contours


# def find_optimal_components_subset(contours, edges):
#     '''Find a crop which strikes a good balance of coverage/compactness.
#     Returns an (x1, y1, x2, y2) tuple.
#     '''
#     c_info = props_for_contours(contours, edges)
#     c_info.sort(key=lambda x: -x['sum'])
#     total = np.sum(edges) / 255
#     area = edges.shape[0] * edges.shape[1]

#     c = c_info[0]
#     del c_info[0]
#     this_crop = c['x1'], c['y1'], c['x2'], c['y2']
#     crop = this_crop
#     covered_sum = c['sum']

#     while covered_sum < total:
#         changed = False
#         recall = 1.0 * covered_sum / total
#         prec = 1 - 1.0 * crop_area(crop) / area
#         f1 = 2 * (prec * recall / (prec + recall))
#         #print '----'
#         for i, c in enumerate(c_info):
#             this_crop = c['x1'], c['y1'], c['x2'], c['y2']
#             new_crop = union_crops(crop, this_crop)
#             new_sum = covered_sum + c['sum']
#             new_recall = 1.0 * new_sum / total
#             new_prec = 1 - 1.0 * crop_area(new_crop) / area
#             new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)

#             # Add this crop if it improves f1 score,
#             # _or_ it adds 25% of the remaining pixels for <15% crop expansion.
#             # ^^^ very ad-hoc! make this smoother
#             remaining_frac = c['sum'] / (total - covered_sum)
#             print(remaining_frac)
#             new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
#             if new_f1 > f1 or (
#                     remaining_frac > 0.25 and new_area_frac < 0.15):
#                 print ('%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> {}' ).format(
#                         i, covered_sum, new_sum, total, remaining_frac,
#                         crop_area(crop), crop_area(new_crop), area, new_area_frac,
#                         f1, new_f1)
#                 crop = new_crop
#                 covered_sum = new_sum
#                 del c_info[i]
#                 changed = True
#                 break

#         if not changed:
#             break

#     return crop


# def pad_crop(crop, contours, edges, border_contour, pad_px=15):
#     '''Slightly expand the crop to get full contours.
#     This will expand to include any contours it currently intersects, but will
#     not expand past a border.
#     '''
#     bx1, by1, bx2, by2 = 0, 0, edges.shape[0], edges.shape[1]
#     if border_contour is not None and len(border_contour) > 0:
#         c = props_for_contours([border_contour], edges)[0]
#         bx1, by1, bx2, by2 = c['x1'] + 5, c['y1'] + 5, c['x2'] - 5, c['y2'] - 5

#     def crop_in_border(crop):
#         x1, y1, x2, y2 = crop
#         x1 = max(x1 - pad_px, bx1)
#         y1 = max(y1 - pad_px, by1)
#         x2 = min(x2 + pad_px, bx2)
#         y2 = min(y2 + pad_px, by2)
#         return crop
    
#     crop = crop_in_border(crop)

#     c_info = props_for_contours(contours, edges)
#     changed = False
#     for c in c_info:
#         this_crop = c['x1'], c['y1'], c['x2'], c['y2']
#         this_area = crop_area(this_crop)
#         int_area = crop_area(intersect_crops(crop, this_crop))
#         new_crop = crop_in_border(union_crops(crop, this_crop))
#         if 0 < int_area < this_area and crop != new_crop:
#             print ('%s -> %s' % (str(crop), str(new_crop)))
#             changed = True
#             crop = new_crop

#     if changed:
#         return pad_crop(crop, contours, edges, border_contour, pad_px)
#     else:
#         return crop




# def strip_border(edges):
#     contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     borders = find_border(contours, edges)
#     print(borders)

#     # borders = sorted(borders, key=lambda i, x1, y1, x2, y2: (x2 - x1) * (y2 - y1))

#     border_contour = contours[borders[0]]
#     edges = remove_border(border_contour, edges)

#     edges = 255 * (edges > 0).astype(np.uint8)

#     # Remove ~1px borders using a rank filter.
#     maxed_rows = rank_filter(edges, -4, size=(1, 20))
#     maxed_cols = rank_filter(edges, -4, size=(20, 1))
#     debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
#     edges = debordered
#     return edges, borders

# def remove_border(contour, ary):
#     '''Remove everything outside a border contour.'''
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
    # c_im = np.zeros(ary.shape)  # mask
    # r = cv2.minAreaRect(contour)    
    # degs = r[2]
    # if angle_from_right(degs) <= 10.0:
    #     box = cv2.boxPoints(r).astype(np.int64) # contour pts must be int64
    #     cv2.drawContours(c_im, [box], 0, 255, -1)   # draw filled contour in mask
    #     cv2.drawContours(c_im, [box], 0, 0, 4)
    # else:
    #     x1, y1, x2, y2 = cv2.boundingRect(contour)
    #     cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
    #     cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    # ary = np.minimum(c_im, ary)
    # return (ary > 0).astype(np.uint8) *255

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

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)


# def bbox_merge(bbox):
#     noIntersectLoop = False
#     noIntersectMain = False
#     posIndex = 0
#     # keep looping until we have completed a full pass over each rectangle
#     # and checked it does not overlap with any other rectangle
#     while noIntersectMain == False:
#         noIntersectMain = True
#         posIndex = 0
#         # start with the first rectangle in the list, once the first 
#         # rectangle has been unioned with every other rectangle,
#         # repeat for the second until done
#         while posIndex < len(bbox):
#             noIntersectLoop = False
#         while noIntersectLoop == False and len(bbox) > 1:
#             a = bbox[posIndex]
#             listBoxes = np.delete(bbox, posIndex, 0)
#             index = 0
#             for b in listBoxes:
#                 #if there is an intersection, the boxes overlap
#                 if intersection(a, b): 
#                     newBox = union(a,b)
#                     listBoxes[index] = newBox
#                     boxes = listBoxes
#                     noIntersectLoop = False
#                     noIntersectMain = False
#                     index = index + 1
#                     break
#                 noIntersectLoop = True
#                 index = index + 1
#         posIndex = posIndex + 1

#     return boxes.astype(np.uint8)

# def bbox_merge(bbox_img):

### Adapted from https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.htmls
def plt_bbox(input_img, num_merge):
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
        Image.fromarray(bbox_img).show()

   
    # Image.fromarray(bbox_img).show()
    
    # bbox_img = cv2.Canny(bbox_img.astype(np.uint8), thresh, thresh*2)
    # contours2, hierarchy2 = cv2.findContours(bbox_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # bbox_img = cv2.drawContours(bbox_img, [c], 0, (0, 255, 0), 2)
    # Image.fromarray(bbox_img).show()

    # bbox_img = cv2.Canny(bbox_img.astype(np.uint8), thresh, thresh*2)
    # contours2, hierarchy2 = cv2.findContours(bbox_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours_poly = [None]*len(contours2)
    # bbox = [None]*len(contours2)
    # for i, c in enumerate(contours2):
    #     contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    #     bbox[i] = cv2.boundingRect(contours_poly[i])    # x, y, w, h
    #     # cv2.drawContours(bbox_img, [c], 0, (255,0,0), 2)
    #     color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
    #     cv2.rectangle(bbox_img, (int(bbox[i][0]), int(bbox[i][1])), \
    #       (int(bbox[i][0]+bbox[i][2]), int(bbox[i][1]+bbox[i][3])), color, -1)   # x, y, x+w, x+h
   
 
    
    # bbox_img = np.zeros((bbox_img.shape[0], bbox_img.shape[1], 3), dtype=np.uint8)
    # for i, c in enumerate(contours):
    #     contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    #     bbox[i] = cv2.boundingRect(contours_poly[i])    # x, y, w, h
    #     cv2.rectangle(bbox_img, (int(bbox[i][0]), int(bbox[i][1])), \
    #       (int(bbox[i][0]+bbox[i][2]), int(bbox[i][1]+bbox[i][3])), (255), -1)   # x, y, x+w, x+h
    
    # bbox_img = cv2.Canny(bbox_img.astype(np.uint8), thresh, thresh*2)
    # contours2, hierarchy2 = cv2.findContours(bbox_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # bbox2 = [None]*len(contours2)
    # for i, c in enumerate(contours2):
    # # get the bounding box of the contour and draw the rect on image
    #     bbox2[i] = cv2.boundingRect(c)
    #     # draw the boundingbox on your image
    #     cv2.drawContours(bbox_img, [c], 0, (255,0,0), 2)
    # bbox_img = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    # for i in range(len(contours)):
    #     color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
    #     cv2.drawContours(bbox_img, contours_poly, i, color)
    #     cv2.rectangle(bbox_img, (int(bbox[i][0]), int(bbox[i][1])), \
    #       (int(bbox[i][0]+bbox[i][2]), int(bbox[i][1]+bbox[i][3])), color, 2)   # x, y, x+w, x+h
  
    # bbox_merged, _ = cv2.groupRectangles(bbox, groupThreshold=1, eps=0.5)
    # bbox_merged = bbox_merge(bbox_img, bbox)

    # hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
    # # For each contour, find the bounding rectangle and draw it
    # for component in zip(contours, hierarchy):
    #     currentContour = component[0]
    #     currentHierarchy = component[1]
        
    #     x,y,w,h = cv2.boundingRect(currentContour)
    #     if currentHierarchy[2] < 0:
    #         # these are the innermost child components
    #         cv2.rectangle(bbox_img, cv2.boundingRect(currentContour), (255, 255, 255), 8)
    #         cv2.rectangle(bbox_img,(x,y),(x+w,y+h),(0,0,255),3)
    #     elif currentHierarchy[3] < 0:
    #         # these are the outermost parent components
    #         cv2.rectangle(bbox_img,(x,y),(x+w,y+h),(0,255,0),3)
    #     Image.fromarray(bbox_img).show()


    # bbox_img = bbox_merge(bbox_img, contours, hierarchy, 0)
    # for i in range(len(bbox_merged)):
    #     cv2.rectangle(bbox_img_full, (int(bbox_merged[i][0]), int(bbox_merged[i][1])), \
    #       (int(bbox_merged[i][0]+bbox_merged[i][2]), int(bbox_merged[i][1]+bbox_merged[i][3])), (255, 255, 255), 8)

    return bbox, bbox_img
    
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


def process_image(path, out_path):
    orig_im = cv2.imread(path)
    scale, img = downscale_image(orig_im)

    # img = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY) # Binarise
    # Image.fromarray(img).show()

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

    bbox, bbox_img = plt_bbox(dilated, num_merge=3)
    bbox_img = Image.fromarray(bbox_img)
    bbox_img.convert('RGB').save('7_bbox.png', 'PNG')
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #remove contour outline
    
    # img = cv2.adaptiveThreshold(img, 
    #                             maxValue=255, 
    #                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                             thresholdType=cv2.THRESH_BINARY, 
    #                             blockSize=115, 
    #                             C=10)   # offset   
    # from skimage.filters import threshold_local
    # T = threshold_local(img, 11, offset = 10, method = 'gaussian')
    # img = (img > T).astype('uint8') * 255


    img = Image.fromarray(img).convert('RGB')   # plot bbox

    img.save('8_final.png', 'PNG')
    # Upscaling
    # a, b = img.size
    # img = img.resize((int(a / scale), int(b / scale)), resample=Image.BICUBIC)  # upscale
    # bbox = [[int(x / scale) for x in box] for box in bbox]

    # Clearing folder
    import shutil
    shutil.rmtree('cropped')
    os.mkdir('cropped') 
    for idx, box in enumerate(bbox):    # Crop the original img
        crop = img.crop([box[0], box[1], box[0]+box[2], box[1]+box[3]])
        print('Saving cropped/{}.png'.format(idx))
        crop.save('cropped/{}.png'.format(idx))

    
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