#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 4:44:15 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Sep 30 2019
###

import os

# for socketio to work with redis (external process)
# import eventlet 
# eventlet.monkey_patch() 

from flask import Flask, url_for
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
# from flask_uploads import UploadSet, IMAGES, configure_uploads
 
# from flask_socketio import SocketIO

from redis import Redis
from rq import Queue, Connection

from server.config import DevelopmentConfig

ROOT_DIR = os.path.dirname(__file__)

# instantiate the extensions
bootstrap = Bootstrap()
db = SQLAlchemy()
migrate = Migrate()
# images = UploadSet('images', IMAGES)
# configure_uploads(app, images)
# socketio = SocketIO()


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
    app.config['IMG_DIR'] = os.path.join(app.static_folder, 'imgs')
    
    # set up extensions
    bootstrap.init_app(app)
    db.init_app(app)
    migrate.init_app(app, db)
    app.redis = Redis.from_url(app.config['REDIS_URL'])
    app.task_queue = Queue(connection=app.redis)
    # socketio.init_app(app, message_queue=app.config['REDIS_URL'])
    # socketio.init_app(app)
    
    # register blueprints
    from server.main import bp as index_bp
    app.register_blueprint(index_bp)

    # shell context for flask cli
    app.shell_context_processor({'app': app})
    
    if not os.path.exists(app.config['DB_FILE']):
        print(app.config['DB_FILE'])
        with app.app_context():
            from server.main import models
            db.create_all()

    return app

from server.main import models