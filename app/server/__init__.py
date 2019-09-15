#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 4:44:15 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sun Sep 15 2019
###

import os

from flask import Flask
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy

import redis
from rq import Queue, Connection

from server.config import DevelopmentConfig
from server.views import index_blueprint

ROOT_DIR = os.path.dirname(__file__)

# instantiate the extensions
bootstrap = Bootstrap()
db = SQLAlchemy()

def create_app(script_info=None):
    # instantiate the app
    app = Flask(
        __name__,
        template_folder='../client/templates',
        static_folder='../client/static'
    )

    # set config
    # app_settings = os.getenv('APP_SETTINGS')
    app_settings = DevelopmentConfig
    app.config.from_object(app_settings)
    app.config['IMG_DIR'] = os.listdir(os.path.join(app.static_folder, 'imgs'))
    
    # set up extensions
    bootstrap.init_app(app)
    db.init_app(app)

    # register blueprints
    app.register_blueprint(index_blueprint)

    # shell context for flask cli
    app.shell_context_processor({'app': app})

    return app

