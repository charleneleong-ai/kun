#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, September 26th 2019, 10:17:11 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Thu Sep 26 2019
###

import argparse
from enum import Enum
import io

from google.cloud import vision
from google.cloud.vision import types
from google.cloud.vision_v1.types import BoundingPoly
from PIL import Image, ImageDraw

use_redis = False
try:
    import redis

    try:
        cache = redis.Redis(host='localhost', port=6379, db=0)
        # execute a command to test connection
        cache.client_list()
        use_redis = True
    except redis.exceptions.ConnectionError:
        pass
except ImportError:
    pass


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


def get_document_bounds(document, feature):
    """Returns document bounds given an image."""

    bounds = []

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if feature == FeatureType.SYMBOL:
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
    return bounds


def render_doc_text(directory):
    from os import listdir, path

    dir_content = [f for f in listdir(directory) if not '_bounded' in f]
    files = [f for f in dir_content if path.isfile(path.join(directory, f))]
    for file_name in files:
        file = path.join(directory, file_name)
        client = vision.ImageAnnotatorClient()
        with io.open(file, 'rb') as image_file:
            content = image_file.read()
        target_bound_str = cache.get(file) if use_redis else None

        if target_bound_str:
            target_bound = BoundingPoly().FromString(target_bound_str)
        else:
            # noinspection PyUnresolvedReferences
            image_type = types.Image(content=content)
            response = client.document_text_detection(image=image_type)
            document = response.full_text_annotation

            bounds = get_document_bounds(document, FeatureType.PARA)

            if len(bounds):
                target_bound = bounds[-1]
                target_bound_str = target_bound.SerializeToString()

                if use_redis:
                    cache.set(file, target_bound_str)

        # target_bound_str exists implies target_bound exists
        if target_bound_str:
            image = Image.open(file)
            # noinspection PyUnboundLocalVariable
            draw_boxes(image, [target_bound], 'green')
            sans_ext, ext = path.splitext(file)
            out_name = sans_ext + '_bounded' + ext
            image.save(out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory where image files are.')
    args = parser.parse_args()
    parser = argparse.ArgumentParser()

    render_doc_text(args.directory)