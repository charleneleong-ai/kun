#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Friday, October 4th 2019, 6:12:59 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Oct 04 2019
###
#!/usr/bin/env python
# encoding: utf-8

import io
import math
import requests
from PIL import Image
 
class ImageOptimAPI():

  endpoint = 'https://im2.io'

  def __init__(self, username):
    self.username = username

  def parse_options(self, options):
    if options:
      opList = []
      try:
        size = options.pop('size')
      except KeyError:
        pass
      else:
        opList.append(size)
      for k, v in options.items():
        if (type(v) == bool):
          opList.append(k)
        else:
          opList.append('%s=%s' % (k, v))
      return ','.join(opList)
    else:
      return 'full'

  def image_from_url(self, file_url, options={}):
    url = self._url(options, file_url)
    return self._call(url)

  def image_from_file(self, file, options={}, resize_pow2=False):
    # Check we have a file object to work with
    if (not isinstance(file, io.IOBase)):
      raise IOError('Image file is not a readable file object')
    # Do we want to resize to the nearest power of 2?
    # This speeds up rendering when using THREE.WebGLRenderer
    if (resize_pow2):
      # Cannot set resize to power of 2 & specific dimensions
      # We could in the future resize the dimensions to power 2
      if 'size' in options:
        raise IOError('Cannot specify dimensions with power of 2 resizing')
      # Load the image to check the dimensions (using name as operating on the
      # file object affects the file object itself)
      img = Image.open(file.name)
      dimensions = {
        'width': img.width,
        'height': img.height,
      }
      # Does the image require resizing?
      resize = False
      for dimension, value in dimensions.items():
        if not self.is_power2(value):
          resize = True
          dimensions[dimension] = self.resize_power2(value)

      # If this requires a resize, change the size option to power of 2 sizes
      if resize:
        options['size'] = '{width}x{height}'.format(
          width=dimensions['width'],
          height=dimensions['height']
        )
        # Allow upscaling of image
        options['fit'] = True
    url = self._url(options)
    return self._call(url, files={'file': file})

  def image_from_file_path(self, file_path, options={}, resize_pow2=False):
    f = open(file_path, 'rb')
    return self.image_from_file(f, options, resize_pow2)

  def _url(self, options, file_url=None):
    # Helper function - build URL from parts
    url_parts = [
      self.endpoint,
      self.username,
      self.parse_options(options)
    ]
    if (file_url):
      url_parts.append(file_url)
    return '/'.join(url_parts)

  @staticmethod
  def _call(url, **kwargs):
    r = requests.post(url, **kwargs)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))

  @staticmethod
  def is_power2(x):
    # Is a number a power of 2
    return ((x & (x - 1)) == 0) and x != 0

  @staticmethod
  def resize_power2(x):
    # Resize a number to nearest greater power of 2
    return int(pow(2, math.ceil(math.log(x, 2))))


import os
import argparse
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory where image files are.')
    args = parser.parse_args()
    parser = argparse.ArgumentParser()

    files = glob.glob(args.directory+'/*.png')
    compressed_dir = args.directory.replace('_output', '_output_compressed')
    if not os.path.exists(compressed_dir):
        os.makedirs(compressed_dir)

    for path in files:
        api = ImageOptimAPI('vfwcvzgsdq')
        img = api.image_from_file_path(path)
        save_path = path.replace('_output', '_output_compressed')
        print(save_path)
        img.save(save_path)