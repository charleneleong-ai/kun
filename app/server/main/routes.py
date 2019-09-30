#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 6:35:09 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Tue Oct 01 2019
###

# project/server/main/routes.py

import os
from datetime import datetime

from flask import render_template, Blueprint, jsonify, \
    url_for, request, redirect, current_app, session

from server.main import bp
from server.main.tasks import train, upload, upload_MNIST, cluster, som
from server.main.models import Image, ImageGrid, clear_tables


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


@bp.route('/', methods=['GET']) 
def home():
    if 'label' not in session:
        session['label'] = ''
        session['num_seen'] = ''
    if 'img_grd_paths' not in session:
        session['img_grd_paths'] = []
        session['img_idx'] = []
        
    return render_template('/shuffle.html', 
            label=session['label'], num_seen = session['num_seen'],     
            imgs=zip(session['img_grd_paths'], session['img_idx']))
    
    
@bp.route('/tasks/<task_type>', methods=['POST'])
def run_task(task_type):
    if task_type=='upload':
        #TODO: Allow for browser upload
        session['label'] = 0
        task_data = session['label']
        # task = current_app.task_queue.enqueue(upload_MNIST, session['label'])   
        img_dir = '../kun_preprocess/char_output/digits/'+ str(session['label']) +'/'
        task = current_app.task_queue.enqueue(upload, (session['label'], img_dir))   
    if task_type=='cluster':
        # session['c_label'] = 8
        # task_data = session['c_label']
        task = current_app.task_queue.enqueue(cluster, job_timeout=180)
        
    if task_type=='som':
        img_idx = [img.idx for img in Image.query.filter_by(processed=False).filter_by(c_label=0).all()]
        c_labels = [img.c_label for img in Image.query.all()]
        task = current_app.task_queue.enqueue(som, (img_idx, c_labels), job_timeout=180)
    
    response_object = {
        'status': 'success',
        'data': {
            'task_type': task_type,
            'task_id': task.get_id(),
            'task_data': task_data
        }
    }
    return jsonify(response_object), 202


@bp.route('/tasks/<task_type>/<task_id>/<task_data>', methods=['GET'])
def get_status(task_type, task_id, task_data):
    task = current_app.task_queue.fetch_job(task_id)
    
    #TODO: create separate routes instead
    if task_type=='upload' and task.get_status()=='finished':
        dataset = task.result   # Uploaded dataset 
        print(dataset)
        task = current_app.task_queue.enqueue(train, dataset, job_timeout=300)
        task_type = 'train'
    elif task_type=='train' and task.get_status()=='finished':
        task_data = task.result     # Gives output_dir for train
        
        task = current_app.task_queue.enqueue(cluster, job_timeout=300)
        task_type = 'cluster'
    elif task_type=='cluster' and task.get_status()=='finished':
        feat, c_labels = task.result
        task_data = set(c_labels)   # Returning num clusters
        print(len(c_labels), len(feat))
        save_img_db(c_labels)
        
        img_idx = [img.idx for img in Image.query.filter_by(processed=False).filter_by(c_label=0).all()]
        task = current_app.task_queue.enqueue(som, (img_idx, c_labels), job_timeout=300)

        task_type = 'som'
    elif task_type=='som' and task.get_status()=='finished':
        img_grd_idx = task.result
        img_grd = ImageGrid(img_grd_idx)    # Update img_grd
        
        session['img_grd_paths'] = img_grd.img_paths
        session['img_idx'] = img_grd.img_idx
        num_seen = Image.query.filter_by(processed=False).filter_by(c_label=0).all()
        session['num_seen'] = len(num_seen)

    if task:
        response_object = {
            'status': 'success', 
            'data': {
                'task_type': task_type,
                'task_id': task.get_id(),
                'task_status': task.get_status(),
                'task_data': task_data,
                'task_result': task.ended_at,
            }
        }
    else:
        response_object = {'status': 'error'}
    return jsonify(response_object)



@bp.route('/selected/<selected_img_idx>/<img_grd_idx>/<img_idx>', methods=['POST'])
def selected_imgs(selected_img_idx, img_grd_idx, img_idx):
    selected_img_idx = selected_img_idx.split(',')
    img_grd_idx = img_grd_idx.split(',')
    img_idx = img_idx.split(',')
    
    seen = [Image.query.filter_by(idx=int(idx)).first().seen() for idx in img_idx]
    selected = [Image.query.filter_by(idx=int(idx)).first() for idx in selected_img_idx]
    
    # Get new imgs to replace seen img_grd_idx
    imgs = Image.query.filter_by(processed=False).filter_by(c_label=0).all()
    img_idx = [img.idx for img in imgs]
    c_labels = [img.c_label for img in Image.query.all()]

    task = current_app.task_queue.enqueue(som, (img_idx, c_labels), job_timeout=300)
    session['num_seen'] = len(imgs)
    
    response_object = {
        'status': 'success',
        'data':{
            'task_type': 'som',
            'task_id': task.get_id()
        },
        'img': {
            'selected_img_idx': selected_img_idx,
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


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS