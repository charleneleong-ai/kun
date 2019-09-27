#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Tuesday, September 24th 2019, 8:31:41 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Wed Sep 25 2019
###


import pytesseract
from PIL import Image

print(pytesseract.image_to_string(Image.open('../kraken/cropped/line_0.png'), lang='chi-tra-vert+chi=tra+chi-sim+chi-sim-vert'))
