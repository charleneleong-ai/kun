#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 6:11:05 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sun Oct 06 2019
###

# server/config.py

import os
import glob

ROOT_DIR = os.path.dirname(__file__)

class BaseConfig(object):
    """Base configuration."""
    WTF_CSRF_ENABLED = True
    SECRET_KEY = b'_5#y2L"F4Q8z\n\xeasdf]'   # for session vars
    DB_FILE = os.path.join(ROOT_DIR, 'imgs.db')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(DB_FILE)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    REDIS_URL = os.getenv('REDIS_URL') or 'redis://'
    QUEUES = ['default']
    MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, 'model', 'output')
    DATASET_DIR = os.path.join(ROOT_DIR, 'datasets')
    UPLOAD_DIR = os.path.join(ROOT_DIR, 'uploads')
    
    OUTPUT_DIR = max(glob.iglob(os.path.join(MODEL_OUTPUT_DIR, 'ae*')), key=os.path.getctime)

        
class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    WTF_CSRF_ENABLED = False

class TestingConfig(BaseConfig):
    """Testing configuration."""
    TESTING = True
    WTF_CSRF_ENABLED = False
    PRESERVE_CONTEXT_ON_EXCEPTION = False
