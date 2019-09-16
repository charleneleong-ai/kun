#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Monday, September 16th 2019, 11:05:27 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Sep 16 2019
###

from flask import Blueprint

bp = Blueprint('index', __name__)

from server.main import routes