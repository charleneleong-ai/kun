#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 6:35:09 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Sep 16 2019
###

# project/server/main/views.py


import redis
from rq import Queue, Connection
from flask import render_template, Blueprint, jsonify, \
    request, redirect,current_app, session

from server.worker import conn
from server.main import bp
from server.main.tasks import train, filtered_MNIST, process_imgs

@bp.route('/', methods=['GET'])
def home():
    jobs = current_app.task_queue.jobs  # Get a list of jobs in the queue
    return render_template('/home.html')


@bp.route('/tasks/<task_type>', methods=['POST'])
def run_task(task_type):
    if task_type=='upload':
        task = current_app.task_queue.enqueue(filtered_MNIST, 8)
    # if task_type=='train':
    #     task==current_app.task_queue.enqueue(train, dataset, job_timeout=300)
    response_object = {
        'status': 'success',
        'data': {
            'task_type': task_type,
            'task_id': task.get_id()
        }
    }
    # session['task'] = task.get_id()
    return jsonify(response_object), 202


@bp.route('/tasks/<task_type>/<task_id>', methods=['GET'])
def get_status(task_type, task_id):

    task = current_app.task_queue.fetch_job(task_id)
    
    #TODO: create seperate routes instead
    if task_type=='upload' and task.get_status()=='finished':
        task_type = 'train'
        dataset = task.result   #TODO: Save to output folder
        task = current_app.task_queue.enqueue(train, dataset, job_timeout=300)
    elif task_type=='train' and task.get_status()=='finished':
        task_type = 'process_imgs'
        task = current_app.task_queue.enqueue(process_imgs, job_timeout=300)
        
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
