#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 6:11:05 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sun Sep 15 2019
###

# server/config.py

import os
basedir = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(__file__)
DB_FILE = 'sqlite:///{}'.format(os.path.join(ROOT_DIR, 'images.db'))

class BaseConfig(object):
    """Base configuration."""
    WTF_CSRF_ENABLED = True
    redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
    REDIS_URL = 'redis://redis:6379/0'
    QUEUES = ['default']
    SQLALCHEMY_DATABASE_URI = DB_FILE
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    WTF_CSRF_ENABLED = False

class TestingConfig(BaseConfig):
    """Testing configuration."""
    TESTING = True
    WTF_CSRF_ENABLED = False
    PRESERVE_CONTEXT_ON_EXCEPTION = False
