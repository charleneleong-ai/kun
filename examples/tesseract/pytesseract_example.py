#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Tuesday, April 23rd 2019, 3:16:10 pm
# Author: Charlene Leong
# -----
# Last Modified: Tuesday, April 23rd 2019, 5:59:17 pm
# Modified By: Charlene Leong
# -----
# Copyright (c) 2019 Deloitte NZ
###

from PIL import Image
import pytesseract
import argparse
import cv2
import os
import glob

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())

# get raw_filename
raw_filename = os.path.basename(os.path.normpath(args["image"]))

# load the example image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# # make a check to see if median blurring should be done to remove
# # noise
# elif args["preprocess"] == "blur":
# 	gray = cv2.medianBlur(gray, 3)

cv2.imshow('image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it - convert to PNG for size
thresholded_filename = "imgs/thresholded/{}-thresholded.png".format(raw_filename.split('.')[0])
cv2.imwrite(thresholded_filename, gray)