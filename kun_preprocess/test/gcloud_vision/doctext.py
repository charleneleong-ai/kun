#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, September 25th 2019, 12:53:40 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sun Oct 13 2019
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
import os
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
        # print(bound.vertices)
        # draw.line(bound.vertices, fill=color, width=3)
        # for point in bound.vertices:
        #     draw.ellipse((point[0] - 4, point[1] - 4, point[0]  + 4, point[1] + 4), fill=color])
        x1 = bound.vertices[0].x  # Adding a little padding
        y1 = bound.vertices[0].y
        x2 = bound.vertices[2].x
        y2 = bound.vertices[2].y
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # draw.polygon([bound.vertices[0].x-10, bound.vertices[0].y-20,
        #                 bound.vertices[1].x+10, bound.vertices[1].y-20,
        #                 bound.vertices[2].x+10, bound.vertices[2].y+10,
        #                 bound.vertices[3].x-10, bound.vertices[3].y+10], None, color)

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
    symbols = []
    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if (feature == FeatureType.SYMBOL):
                            # if feature in string.printable:
                            #     continue
                            symbols.append(symbol)
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
    return bounds, symbols

import string

def render_doc_text(filein, fileout):
    image = Image.open(filein)
    bounds, _ = get_document_bounds(filein, FeatureType.PAGE)
    draw_boxes(image, bounds, 'blue')
    bounds, _ = get_document_bounds(filein, FeatureType.PARA)
    draw_boxes(image, bounds, 'red')
    bounds, _ = get_document_bounds(filein, FeatureType.WORD)
    draw_boxes(image, bounds, 'green')
    bounds, symbols = get_document_bounds(filein, FeatureType.SYMBOL)
    draw_boxes(image, bounds, 'purple')

    # symbol = property {
    #         detected_languages {
    #             language_code: "en"
    #         }
    #         detected_break {
    #             type: LINE_BREAK
    #         }
    #         }
    #         bounding_box {
    #         vertices {
    #             x: 427
    #             y: 1740
    #         }
    #         vertices {
    #             x: 432
    #             y: 1740
    #         }
    #         vertices {
    #             x: 432
    #             y: 1761
    #         }
    #         vertices {
    #             x: 427
    #             y: 1761
    #         }
    #         }
    #         text: "."
    #         confidence: 0.9599999785423279}


    # # Cropping the image and saving the symbol
    # bbox_img = Image.open(filein).convert('RGB')   
    # bbox_draw = ImageDraw.Draw(bbox_img)
    # for idx, s in enumerate(symbols):
    #     # if s.confidence < 0.7:  continue
    #     # if s.property.detected_break != None: # First check if break
    #     #     char_type = str(s.property.detected_break.type)
    #     # elif s.property.detected_languages[0]!=None:    ### Not reliable, sometimes absent
    #     #     char_type  = s.property.detected_languages[0].language_code
    #     if s.text in string.ascii_letters:
    #         char_type = 'en'
    #     elif s.text in string.digits:
    #         char_type = 'digits'
    #     elif s.text in string.punctuation:
    #         char_type = 'punctuation'
    #     else:
    #         char_type = 'zh'
    #     print(s.text)
    #     bbox = s.bounding_box
    #     x1 = bbox.vertices[0].x  # Adding a little padding
    #     y1 = bbox.vertices[0].y
    #     x2 = bbox.vertices[2].x
    #     y2 = bbox.vertices[2].y
    #     w = (x2-x1)
    #     h = (y2-y1)
    #     # if (w*h) < 784: continue    # If smaller than MNIST ~(28*28)
            
    #     if not os.path.exists('char_output/{}/{}'.format(char_type, s.text)):
    #         os.makedirs('char_output/{}/{}'.format(char_type, s.text))

    #     bbox_draw.rectangle([x1, y1, x2, y2], outline='purple', width=5)
        # crop = image.crop([x1, y1, x2, y2])
        # crop.save('char_output/{}/{}/{}_{}_{}_{}.png'.format(char_type, s.text, x1, y1, x2, y2))

    # #     crop.save('char_output/')
  
    if fileout is not 0:
        image.save(fileout)
    else:
        image.show()



if __name__ == '__main__':
    # [START vision_document_text_tutorial_run_application]
    parser = argparse.ArgumentParser()
    parser.add_argument('detect_file', help='The image for text detection.')
    parser.add_argument('--out_file', help='Optional output file', default=0)
    args = parser.parse_args()

    render_doc_text(args.detect_file, args.out_file)
    # [END vision_document_text_tutorial_run_application]
# [END vision_document_text_tutorial]