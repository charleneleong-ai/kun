#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 6:35:09 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sun Sep 15 2019
###

# project/server/main/views.py


import redis
from rq import Queue, Connection
from flask import render_template, Blueprint, jsonify, \
    request, current_app

from server.worker import conn
from server.tasks import train, filtered_MNIST

index_blueprint = Blueprint('index', __name__)
r = redis.Redis()
q = Queue(connection=r)

@index_blueprint.route('/', methods=['GET'])
def home():
    jobs = q.jobs  # Get a list of jobs in the queue
    return render_template('/home.html')


@index_blueprint.route('/tasks/<task_type>', methods=['POST'])
def run_task(task_type):
    if task_type=='upload':
        task = q.enqueue(filtered_MNIST, 8)
        
    response_object = {
        'status': 'success',
        'data': {
            'task_type': task_type,
            'task_id': task.get_id()
        }
    }
    return jsonify(response_object), 202


@index_blueprint.route('/tasks/<task_type>/<task_id>', methods=['GET'])
def get_status(task_type, task_id):

    task = q.fetch_job(task_id)

    if task_type=='upload' and task.get_status()=='finished':
        task_type = 'train'
        dataset = task.result
        task = q.enqueue(train, dataset, job_timeout=300)
        
    if task:
        response_object = {
            'status': 'success',
            'data': {
                'task_type': task_type,
                'task_id': task.get_id(),
                'task_status': task.get_status(),
                'task_result': task.ended_at,
            }
        }
    else:
        response_object = {'status': 'error'}
    return jsonify(response_object)
