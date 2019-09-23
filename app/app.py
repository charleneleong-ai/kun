#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, September 5th 2019, 9:14:26 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Sep 23 2019
###

# from server import create_app, socketio
from server import create_app


HOST = '0.0.0.0'
PORT = 8888

app = create_app()

# TODO: Remove later, for dev
# from werkzeug.debug import DebuggedApplication
# app.debug = True
# app.wsgi_app = DebuggedApplication(app.wsgi_app, evalex=True)

# import redis
# # @cli.command('run_worker')
# def run_worker():
#     redis_url = app.config['REDIS_URL']
#     redis_connection = redis.from_url(redis_url)
#     with Connection(redis_connection):
#         worker = Worker(app.config['QUEUES'])
#         worker.work()


if __name__ == '__main__':
    # run_worker()
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT,
            use_reloader=True)
    # socketio.run(app, host=HOST, port=PORT, debug=True,  use_reloader=True)