#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, September 25th 2019, 12:53:40 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Thu Sep 26 2019
###


"""Outlines document text given an image.
Example:
    python doctext.py resources/text_menu.jpg
"""
# [START vision_document_text_tutorial]
# [START vision_document_text_tutorial_imports]
import argparse
from enum import Enum
import io

from google.cloud import vision
from google.cloud.vision import types
from google.cloud.vision_v1.types import BoundingPoly
from PIL import Image, ImageDraw
# [END vision_document_text_tutorial_imports]
from io import BytesIO

def convertToJpeg(im):
    with BytesIO() as f:
        im.save(f, format='JPEG')
        return f.getvalue()
        
def convertToPNG(im):
    with BytesIO() as f:
        im.save(f, format='PNG')
        f.seek(0)
        return Image.open(f)

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def draw_boxes(image, bounds, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        draw.polygon([
            bound.vertices[0].x - 10, bound.vertices[0].y,
            bound.vertices[1].x + 10, bound.vertices[1].y,
            bound.vertices[2].x + 5, bound.vertices[2].y,
            bound.vertices[3].x, bound.vertices[3].y], None, color)
    return image


def get_document_bounds(image_file, feature):
    # [START vision_document_text_tutorial_detect_bounds]
    """Returns document bounds given an image."""
    client = vision.ImageAnnotatorClient()

    bounds = []

    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.document_text_detection(image=image)
    document = response.full_text_annotation
    texts = response.text_annotations

    # Save text output
    with open('out.txt', 'w') as f: 
        for text in texts:
            print('"{}"\n'.format(text.description))
            f.write('"{}\n"'.format(text.description)) 

    # import string

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if (feature == FeatureType.SYMBOL):
                            # if feature in string.printable:
                            #     continue
                            bounds.append(symbol.bounding_box)
                
                    if (feature == FeatureType.WORD):
                        bounds.append(word.bounding_box)
                if (feature == FeatureType.PARA):
                    bounds.append(paragraph.bounding_box)

            if (feature == FeatureType.BLOCK):
                bounds.append(block.bounding_box)

        if (feature == FeatureType.PAGE):
            bounds.append(block.bounding_box)

    # The list `bounds` contains the coordinates of the bounding boxes.
    # [END vision_document_text_tutorial_detect_bounds]
    return bounds


def render_doc_text(filein, fileout):
    image = Image.open(filein)
    # image = convertToJpeg(image)
    bounds = get_document_bounds(filein, FeatureType.PAGE)
    draw_boxes(image, bounds, 'blue')
    bounds = get_document_bounds(filein, FeatureType.PARA)
    draw_boxes(image, bounds, 'red')
    bounds = get_document_bounds(filein, FeatureType.WORD)
    draw_boxes(image, bounds, 'yellow')
    bounds = get_document_bounds(filein, FeatureType.SYMBOL)
    draw_boxes(image, bounds, 'green')

    if fileout is not 0:
        image.save(fileout)
    else:
        image.show()


if __name__ == '__main__':
    # [START vision_document_text_tutorial_run_application]
    parser = argparse.ArgumentParser()
    parser.add_argument('detect_file', help='The image for text detection.')
    parser.add_argument('-out_file', help='Optional output file', default=0)
    args = parser.parse_args()

    render_doc_text(args.detect_file, args.out_file)
    # [END vision_document_text_tutorial_run_application]
# [END vision_document_text_tutorial]