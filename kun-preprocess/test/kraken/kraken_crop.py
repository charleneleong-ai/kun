#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Tuesday, September 24th 2019, 8:31:41 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Wed Sep 25 2019
###


from PIL import Image
import pprint
import codecs, json

img = Image.open('bw.tif')

line_crops = json.loads((codecs.open('lines.json', 'r', encoding='utf-8').read()))
pprint.pprint(line_crops)
for idx, box in enumerate(line_crops['boxes']):
    img = img.crop((box[0], box[1], box[2], box[3]))
    img.save('cropped/line_{}.png'.format(idx))