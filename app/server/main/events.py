#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 22nd 2019, 8:01:29 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Sep 23 2019
###

from flask import session
from flask_socketio import emit
from server import socketio

def connect():
    print('CONNECT\n\n\n')
    emit('after connect', {'data':'Hello'})

@socketio.on('connect')
def test_connect():
    print('connect')
    emit('after connect', {'data':'Hello'})
    # emit('task',{})

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
    emit('my response', json, callback=messageReceived)

# @socketio.on('task')
# def task(task_type, task_id):
#     emit('status', {'status':'cheese'})
# @socketio.on('upload')
# def upload():
#     task = current_app.task_queue.enqueue(filtered_MNIST, 8)     

socketio.on_event('connect', connect)
