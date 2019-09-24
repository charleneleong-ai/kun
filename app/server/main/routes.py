#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 6:35:09 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Tue Sep 24 2019
###

# project/server/main/routes.py

import os
from datetime import datetime

from flask import render_template, Blueprint, jsonify, \
    url_for, request, redirect, current_app, session

from server.main import bp
from server.main.tasks import train, filtered_MNIST, cluster, som
from server.main.models import Image, ImageGrid, clear_tables

@bp.route('/', methods=['GET'])
def home():
    if 'img_grd_paths' not in session:
        session['img_grd_paths'] = []
        session['img_idx'] = []
    return render_template('/shuffle.html', imgs=zip(session['img_grd_paths'], session['img_idx']))
    
    
@bp.route('/tasks/<task_type>', methods=['POST'])
def run_task(task_type):
    if task_type=='upload':
        #TODO: Allow for browser upload
        session['label'] = 8
        task = current_app.task_queue.enqueue(filtered_MNIST, session['label'])     
    if task_type=='cluster':
        task = current_app.task_queue.enqueue(cluster, job_timeout=180)
       
    if task_type=='som':
        label_0_idx = [img.idx for img in Image.query.filter_by(processed=False).filter_by(c_label=0).all()]
        c_labels = [img.c_label for img in Image.query.all()]
        task = current_app.task_queue.enqueue(som, (label_0_idx, c_labels), job_timeout=180)
    
    response_object = {
        'status': 'success',
        'data': {
            'task_type': task_type,
            'task_id': task.get_id()
        }
    }
    return jsonify(response_object), 202


@bp.route('/tasks/<task_type>/<task_id>', methods=['GET'])
def get_status(task_type, task_id):
    task = current_app.task_queue.fetch_job(task_id)
    task_data = ''
    #TODO: create separate routes instead
    if task_type=='upload': 
        task_data = session['label']
    elif task_type=='upload' and task.get_status()=='finished':
        task_type = 'train'
        dataset = task.result   
        task = current_app.task_queue.enqueue(train, dataset, job_timeout=300)
        task_data = task.result
    elif task_type=='train' and task.get_status()=='finished':
        task_type = 'cluster'
        task = current_app.task_queue.enqueue(cluster, job_timeout=300)
        
    elif task_type=='cluster' and task.get_status()=='finished':
        task_type = 'som'
        feat, c_labels = task.result
        print(len(c_labels), len(feat))
        save_img_db(c_labels)
        
        label_0_idx = [img.idx for img in Image.query.filter_by(processed=False).filter_by(c_label=0).all()]
        task = current_app.task_queue.enqueue(som, (label_0_idx, c_labels), job_timeout=300)

    elif task_type=='som' and task.get_status()=='finished':
        img_grd_idx = task.result
        img_grd = ImageGrid(img_grd_idx)    # Update img_grd
        session['img_grd_paths'] = img_grd.img_paths
        session['img_idx'] = img_grd.img_idx

    if task:
        response_object = {
            'status': 'success', 
            'data': {
                'task_type': task_type,
                'task_id': task.get_id(),
                'task_status': task.get_status(),
                'task_result': task.ended_at,
                'task_data': task_data
            }
        }
    else:
        response_object = {'status': 'error'}
    return jsonify(response_object)



@bp.route('/selected/<img_idx>/<img_grd_idx>', methods=['POST'])
def selected_imgs(img_idx, img_grd_idx):
    img_idx = img_idx.split(',')
    img_grd_idx = img_grd_idx.split(',')
    seen = [Image.query.filter_by(idx=int(idx)).first().seen() for idx in img_idx]
    selected = [Image.query.filter_by(idx=int(idx)).first() for idx in img_idx]

    # Get new imgs to replace seen img_grd_idx
    imgs = Image.query.filter_by(processed=False).filter_by(c_label=0).all()
    label_0_idx = [img.idx for img in imgs]
    c_labels = [img.c_label for img in Image.query.all()]

    task = current_app.task_queue.enqueue(som, (label_0_idx, c_labels), job_timeout=300)

    response_object = {
        'status': 'success',
        'data':{
            'task_type': 'som',
            'task_id': task.get_id()
        },
        'img': {
            'img_idx': img_idx,
            'img_grd_idx': img_grd_idx,
            'seen': seen,
            'num_seen':len(imgs)
        }
    }
    return jsonify(response_object), 202


# TODO: Should be done as tasks but yet to work out how to declare db instance in tasks.py ><
def save_img_db(c_labels):
    img_names = os.listdir(current_app.config['IMG_DIR'])
    img_paths = [url_for('static', filename=os.path.join('imgs', img)) for img in img_names]
    clear_tables()
    for img_path in img_paths:
        idx = int(os.path.basename(os.path.normpath(img_path)).split('.')[0])
        print(idx, img_path, c_labels[idx])
        img = Image(idx=idx, c_label=int(c_labels[idx]), img_path=img_path, processed=False).add()

