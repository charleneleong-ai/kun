#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, September 26th 2019, 10:17:11 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sat Oct 05 2019
###

import sys
import argparse
from enum import Enum
import io
import glob
import os
import numpy as np

from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw
import cv2
import string
import re

from transform import center_img, translate_bbox, square_img, resize, make_square
from utils import find_contours, find_bbox, add_padding_img, add_padding_bbox

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


# def draw_boxes(image, bounds, color):
#     """Draw a border around the image using the hints in the vector list."""
#     draw = ImageDraw.Draw(image)

#     for bound in bounds:
#         draw.polygon([bound.vertices[0].x-10, bound.vertices[0].y-20,
#                       bound.vertices[1].x+10, bound.vertices[1].y-20,
#                       bound.vertices[2].x+10, bound.vertices[2].y+10,
#                       bound.vertices[3].x-10, bound.vertices[3].y+10], None, color)
#     return image

def zh_detect(texts):
    # chinese
    if re.search("[\u4e00-\u9FFF]", texts):
        return True
    return False

def get_document_bounds(document, feature):
    """Returns document bounds given an image."""
    # Collect specified feature bounds by enumerating all document features
    bounds = []
    symbols = []
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if feature == FeatureType.SYMBOL:
                            symbols.append(symbol)
                            bounds.append(symbol.bounding_box)

                    if feature == FeatureType.WORD:
                        bounds.append(word.bounding_box)

                if feature == FeatureType.PARA:
                    bounds.append(paragraph.bounding_box)

            if feature == FeatureType.BLOCK:
                bounds.append(block.bounding_box)

        if feature == FeatureType.PAGE:
            # noinspection PyUnboundLocalVariable
            bounds.append(block.bounding_box)

    # The list `bounds` contains the coordinates of the bounding boxes.
    return bounds, symbols


def crop_symbols(file, symbols, img):
    bbox_img = Image.open(file).convert('RGB')
    bbox_draw = ImageDraw.Draw(bbox_img)

    # symbols = [symbols[230 ]]
    for idx, s in enumerate(symbols):
        if s.confidence < 0.5:  
            print('Confidence < 0.50') 
            continue
 
        x1 = s.bounding_box.vertices[0].x-20  # Adding a little padding
        y1 = s.bounding_box.vertices[0].y-20
        x2 = s.bounding_box.vertices[2].x+10
        y2 = s.bounding_box.vertices[2].y+10
        w = (x2-x1)
        h = (y2-y1)
        if (w*h) < 784: 
            print('Size smaller than (28*28)') 
            continue    # If smaller than MNIST ~(28*28)
               
        crop = img.crop([x1, y1, x2, y2])
        # if idx % 500 == 0:
        #     crop.show()

        contours, hierarchy = find_contours(crop)
        if len(contours) == 0:
            print('No contours found')
            continue
        
        b_x, b_y, b_w, b_h = find_bbox(crop)
        if (b_w*b_h) < 0.1*(w*h):  # If the bbox area <10%
            print('Bbox area < 0.10 img area')
            continue 

        if s.text in string.ascii_letters:
            char_type = 'en'
        elif s.text in string.digits:
            char_type = 'digits'
        elif s.text in string.punctuation:
            char_type = 'punctuation'
            if s.text=='.':
                s.text='period'
            elif s.text=='/':
                s.text='backslash'
            elif s.text==':':
                s.text='colon'
        elif zh_detect(s.text):
            char_type = 'zh'
        else:
            print('Not valid character')
            continue
  
        ## Correcting bbox crop
        if char_type == 'punctuation':  # Punctuation bbox is small, so make square img
            img = square_img(img)   
        elif char_type == 'zh':         # Pad zh chars
            # Adding padding if cut off
            crop, img_crop, bbox = add_padding_img(img, [b_x, b_y, b_w, b_h], [x1, y1, w, h], padding=20)
            x1, y1, w, h = img_crop
            b_x, b_y, b_w, b_h = bbox

            # bbox_img = cv2.rectangle(np.asarray(crop), (int(b_x), int(b_y)), (int(b_x+b_w), int(b_y+b_h)), 255, -1)   # x, y, x+w, y+h
            # if idx % 100 == 0:
            #   Image.fromarray(bbox_img).show()
            
            # Centering character
            x1, y1, w, h = center_img([b_x, b_y,b_w, b_h], img_crop)  
            x1, y1, w, h = make_square([x1, y1, w, h]) 
            if x1 < 0 or y1 < 0:
                print('Invalid coord')
                continue
            
        else:   # Grow bbox for en and digits
            x1, y1, w, h = add_padding_bbox([b_x, b_y, b_w, b_h], [x1, y1, w, h], padding=20)
            x1, y1, w, h = make_square([x1, y1, w, h]) 
            if x1 < 0 or y1 < 0:
                print('Invalid coord')
                continue

        crop = img.crop([x1, y1, x1+w, y1+h])
        crop = resize(np.asarray(crop), width=28, height=28)
        crop = Image.fromarray(crop)

        contours, hierarchy = find_contours(crop)   # Last contour check
        if len(contours) == 0:
            print('No contours found')
            continue
        
        if not os.path.exists('char_output/{}/{}'.format(char_type, s.text)):
            os.makedirs('char_output/{}/{}'.format(char_type, s.text))
    
        fname = os.path.basename(os.path.normpath(file))
        fname, ext = os.path.splitext(fname)
        print('Saving {}/{}/{:.2f}_{}_{}.{}.{}.{}.png'.format(
            char_type, s.text, s.confidence*100, fname, x1, y1, x2, y2))
        crop.save('char_output/{}/{}/{:.2f}_{}_{}.{}.{}.{}.png'.format(
            char_type, s.text, s.confidence*100, fname, x1, y1, x2, y2))

        bbox_draw.rectangle([x1, y1, x1+w, y1+h], outline='purple', width=5)
            
    return bbox_img

def render_doc_text(file, fileout):
    client = vision.ImageAnnotatorClient()
    with io.open(file, 'rb') as image_file:
        content = image_file.read()

    image_type = types.Image(content=content)
    response = client.document_text_detection(image=image_type)
    if response == None:    # No response
        print('No response from API')
        raise ValueError('No response from API')
    document = response.full_text_annotation

    bounds, symbols = get_document_bounds(document, FeatureType.SYMBOL)
    if len(bounds)==0:   # No bounds
        print('No bounds found') 
        raise ValueError('No bounds found')

    img = Image.open(file)
    img = crop_symbols(file, symbols, img)
    
    print('Saving {}\n'.format(fileout))
    img.save(fileout)

    # Save text output
    texts = response.text_annotations
    # text = texts[0] # First text is full page 
    fname = file.replace('processed', 'bounded')   # Replace image with txt
    fname = fname.replace('.png', '.txt')   # Replace image with txt
    print('Saving ', fname)
    print('{}'.format(texts[0].description)) 
    for text in texts:
        with open(fname, 'w') as f:
            f.write('{}\n'.format(text.description))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory where image files are.')
    args = parser.parse_args()
    parser = argparse.ArgumentParser()

    files = glob.glob(args.directory+'/*.png')
    
    char_outputdir = 'char_output'
    bounded_outputdir = 'NZCGMJ_bounded'

    if not os.path.exists(char_outputdir):
        os.mkdir(char_outputdir)
    if not os.path.exists(bounded_outputdir):
        os.mkdir(bounded_outputdir)
    # files = ['NZCGMJ_processed/NZCGMJv004i002d19520901f0029p002.png']
    
    for path in files:
        fname = os.path.basename(os.path.normpath(path))
        fname, ext = os.path.splitext(fname)
        fname = fname+'_bounded.png'
        out_path = os.path.join('NZCGMJ_bounded', fname)
        if os.path.exists(out_path):
            continue
        
        # To read error o/p
        try:
            print('Processing ' + path)
            render_doc_text(path, out_path)
        except Exception as e:
            fname = os.path.basename(os.path.normpath(path))
            with open(os.path.join('NZCGMJ_bounded', 'error_files.txt'), 'a') as f:
                f.write('{} {} Error \n'.format(fname, e))
                print('{} {} Error'.format(fname, e))
