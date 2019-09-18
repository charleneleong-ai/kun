#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 6:35:09 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Wed Sep 18 2019
###

# project/server/main/views.py

import os

import redis
from rq import Queue, Connection
from flask import render_template, Blueprint, jsonify, \
    url_for, request, redirect, current_app, session

from server.worker import conn
from server.main import bp
from server.main.tasks import train, filtered_MNIST, cluster
from server.main.models import Image, clear_tables

@bp.route('/', methods=['GET'])
def home():
    img_names = os.listdir(current_app.config['IMG_DIR'])
    imgs = [url_for('static', filename=os.path.join('imgs', img)) for img in img_names]
    return render_template('/shuffle.html', imgs=imgs)


@bp.route('/tasks/<task_type>', methods=['POST'])
def run_task(task_type):
    if task_type=='upload':
        task = current_app.task_queue.enqueue(filtered_MNIST, 8)
    if task_type=='cluster':
        task = current_app.task_queue.enqueue(cluster,  job_timeout=300)
       
    if task_type=='save_imgs':
        print(Image.query.order_by(Image.idx).all())
        # User.query.order_by(User.username).all()

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
        task_type = 'cluster'
        task = current_app.task_queue.enqueue(cluster, job_timeout=300)
    elif task_type=='cluster' and task.get_status()=='finished':
        task_type = 'save_imgs'
        # task = current_app.task_queue.enqueue(process_imgs, job_timeout=300)
        c_labels, feat = task.result
        print(len(c_labels), len(feat))
        save_img_db(c_labels)
        
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

def save_img_db(c_labels):
    img_names = os.listdir(current_app.config['IMG_DIR'])
    img_paths = [url_for('static', filename=os.path.join('imgs', img)) for img in img_names]
    clear_tables()
    for img_path in img_paths:
        idx = int(os.path.basename(os.path.normpath(img_path)).split('.')[0])
        print(idx, img_path, c_labels[idx])
        img_db = Image(idx=idx, c_label=int(c_labels[idx]), img_grd_idx=None, img_path=img_path)
        img_db.save_to_db()
