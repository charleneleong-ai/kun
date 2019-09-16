#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, September 5th 2019, 9:14:26 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Sep 16 2019
###

import warnings
warnings.filterwarnings('ignore')

import glob
import os
from datetime import datetime

from flask import Flask
from flask import render_template, url_for
from flask import request, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy

from server import create_app

import redis
from rq import Connection, Worker

HOST = '0.0.0.0'
PORT = 8888

app = create_app()

# @cli.command('run_worker')
def run_worker():
    redis_url = app.config['REDIS_URL']
    redis_connection = redis.from_url(redis_url)
    with Connection(redis_connection):
        worker = Worker(app.config['QUEUES'])
        worker.work()


if __name__ == '__main__':
    # run_worker()
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)
