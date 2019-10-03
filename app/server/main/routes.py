#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 6:35:09 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Thu Oct 03 2019
###

# project/server/main/routes.py

import os
from datetime import datetime

from flask import render_template, Blueprint
from flask import jsonify, json, url_for, request, redirect
from flask import current_app, session

from server.main import bp
from server.main.tasks import train, upload, upload_MNIST, cluster, som
from server.main.models import Image, ImageGrid, clear_tables

import pprint
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


@bp.route('/', methods=['GET']) 
def home():
    if 'LABEL' not in session:
        session['LABEL'] = ''
        
    if 'img_grd_paths' not in session:
        session['img_grd_paths'] = []
        session['img_idx'] = []
        session['c_labels'] = []
        session['img_grd_c_labels'] = []
        session['NUM_CLUSTERS'] = 0
        session['NUM_SEEN'] = 0
        session['NUM_FILTERED'] = 0
        session['NUM_REFRESH'] = 0
        
    return render_template('/shuffle.html', 
            LABEL=session['LABEL'], img_grd_c_labels=session['img_grd_c_labels'], NUM_SEEN=session['NUM_SEEN'], 
            NUM_CLUSTERS=session['NUM_CLUSTERS'], NUM_FILTERED=session['NUM_FILTERED'], NUM_REFRESH=session['NUM_REFRESH'],
            imgs=zip(session['img_grd_paths'], session['img_idx'], session['c_labels']))
    
    
@bp.route('/tasks/<task_type>', methods=['POST'])
def run_task(task_type):
    session['LABEL'] = '中'
    task_data = {'LABEL':  session['LABEL'] }
    
    if task_type=='upload':
        #TODO: Allow for browser upload
        # task = current_app.task_queue.enqueue(upload_MNIST, session['LABEL'])   
        img_dir = '../kun_preprocess/char_output/zh/'+ str(session['LABEL']) +'/'
        task = current_app.task_queue.enqueue(upload, (session['LABEL'], img_dir))   

    if task_type=='cluster': # Rerunning HDBSCAN
        task = current_app.task_queue.enqueue(cluster, session['LABEL'], job_timeout=180)
        session['NUM_CLUSTERS'] = 0
        session['img_grd_c_labels'] = []
        
    if task_type=='som':    # Resetting the SOM
        # Resetting processed in db
        for img in Image.query.filter_by(label=session['LABEL']).filter_by(processed=True):
            img.reset()   
        # Reloading som weights
        if 'output_dir' not in session: 
            session['output_dir'] = current_app.config['OUTPUT_DIR']
        os.remove(os.path.join(session['output_dir'], '_som.json'))  
        
        task = run_som_task()
        
    pprint.pprint(session)
    response_object = {
        'status': 'success',
        'task': {
            'task_type': task_type,
            'task_id': task.get_id(),
            'task_data': task_data
            
        }
    }

    return jsonify(response_object), 202

def run_som_task():
    img_idx = [img.idx for img in Image.query.filter_by(label=session['LABEL'])
                                        .order_by(Image.c_label.desc()).all()]
    c_labels = [img.c_label for img in Image.query.filter_by(label=session['LABEL']).all()]
    task = current_app.task_queue.enqueue(som, (img_idx, c_labels), job_timeout=180)

    # Resetting session vars
    session['NUM_REFRESH'] = 0
    session['NUM_FILTERED'] = 0 
    session['NUM_SEEN'] = len(img_idx)

    return task


@bp.route('/tasks/<task_type>/<task_id>', methods=['POST'])
def get_status(task_type, task_id):
    task = current_app.task_queue.fetch_job(task_id)
    if request.is_json:
        req = request.get_json()
        print(req)
        task_data = req['task_data']
        
    #TODO: create separate routes instead
    if task_type=='upload' and task.get_status()=='finished':
        dataset = task.result   # Uploaded dataset 
        task_data['NUM_IMGS'] = len(dataset)
        task_data['NUM_TRAIN'] = len(dataset.train)
        task_data['NUM_TEST'] = len(dataset.test)

        session['NUM_IMGS'] = len(dataset)
        session['NUM_TRAIN'] = len(dataset.train)
        session['NUM_TEST'] = len(dataset.test)
        print(dataset)
        task = current_app.task_queue.enqueue(train, dataset, job_timeout=600)  # 10 min training
        task_type = 'train'

    elif task_type=='train' and task.get_status()=='finished':
        batch_size, lr, epoch, output_dir = task.result    
        session['ae'] = {'bs': batch_size, 'lr': lr, 'epoch': epoch}
        session['OUTPUT_DIR'] = output_dir
        task = current_app.task_queue.enqueue(cluster, session['LABEL'], job_timeout=300)
        
        session['NUM_CLUSTERS'] = 0
        session['img_grd_c_labels'] = []
        task_type = 'cluster'
        
    elif task_type=='train':    # Report job progress
        task_data.update(task.meta)
        task.refresh()

    elif task_type=='cluster' and task.get_status()=='finished':
        feat, c_labels = task.result

        task_data['NUM_CLUSTERS'] = len(set(c_labels))
        session['NUM_CLUSTERS'] = len(set(c_labels))   
        print(len(c_labels), len(feat))
        
        save_img_db(c_labels)
        task = run_som_task()
        task_type = 'som'
        
    elif task_type=='som' and task.get_status()=='finished':
        img_grd_idx = task.result
        print(img_grd_idx)
        img_grd = ImageGrid(img_grd_idx)    # Update img_grd
        session['img_grd_paths'] = img_grd.img_paths
        session['img_idx'] = img_grd.img_idx
        session['c_labels'] = img_grd.c_labels
        session['img_grd_c_labels'] = list(set(session['c_labels']))
        print(session['img_grd_c_labels'])
        
        task_data['NUM_SEEN'] = session['NUM_SEEN']
        task_data['NUM_FILTERED'] =  session['NUM_FILTERED']
        task_data['NUM_REFRESH'] = session['NUM_REFRESH'] 
        task_data['img_grd_c_labels'] = session['img_grd_c_labels']

    elif task_type=='som':
        task_data.update(task.meta)
        task.refresh()
        
    if task:
        response_object = {
            'status': 'success', 
            'task': {
                'task_type': task_type,
                'task_id': task.get_id(),
                'task_status': task.get_status(),
                'task_data': task_data,
                'task_result': task.ended_at,
            }
        }
    else:
        response_object = {'status': 'error'}
    return jsonify(response_object), 202



@bp.route('/selected/<selected_img_idx>/<img_grd_idx>/<img_idx>', methods=['POST'])
def selected_imgs(selected_img_idx, img_grd_idx, img_idx):
    selected_img_idx = selected_img_idx.split(',')
    img_grd_idx = img_grd_idx.split(',')
    img_idx = img_idx.split(',')
    
    # Setting all imgs in grd to processed
    seen = [Image.query.filter_by(label=session['LABEL'])
                .filter_by(idx=int(idx)).first().seen() for idx in img_idx]
    selected = [Image.query.filter_by(label=session['LABEL'])
                .filter_by(idx=int(idx)).first() for idx in selected_img_idx]
    
    # Get new unprocessed imgs to replace img_grd_idx 
    imgs = Image.query.filter_by(label=session['LABEL']) \
                        .filter_by(processed=False).order_by(Image.c_label.desc()).all()
    img_idx = [img.idx for img in imgs]
    c_labels = [img.c_label for img in Image.query.filter_by(label=session['LABEL']).all()]

    task = current_app.task_queue.enqueue(som, (img_idx, c_labels), job_timeout=300)

    # Updating num imgs seen
    session['NUM_SEEN'] = len(imgs)
    session['NUM_FILTERED'] += len(selected)
    session['NUM_REFRESH'] += 1
    
    task_data = {'LABEL': session['LABEL'],
                'NUM_REFRESH': session['NUM_REFRESH'], 
                'NUM_FILTERED': session['NUM_FILTERED'],
                'NUM_SEEN': session['NUM_SEEN']
                 }

    response_object = {
        'status': 'success',
        'task':{
            'task_type': 'som',
            'task_id': task.get_id(),
            'task_status': task.get_status(),
            'task_data': task_data,
            'task_result': task.ended_at
        },
        'img': {
            'selected_img_idx': selected_img_idx,
            'img_grd_idx': img_grd_idx,
            'seen': seen,
            'NUM_SEEN': session['NUM_SEEN'],
            'NUM_REFRESH': session['NUM_REFRESH']
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
        img = Image(idx=idx, label=session['LABEL'], c_label=int(c_labels[idx]), 
                img_path=img_path, processed=False).add()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
