#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 6:35:09 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Mon Oct 07 2019
###

# app/server/main/routes.py

import os
import glob
import shutil
import pprint
import numpy as np

from flask import render_template, Blueprint
from flask import jsonify, url_for, request, redirect, flash
from flask import current_app, session

from server.main import bp
from server.main.tasks import extract_zip, load_data, load_MNIST, train, cluster, som
from server.main.models import Image, ImageGrid, clear_tables
from server.utils.datasets.imgbucket import ImageBucket

def is_zipfile(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in set(['zip'])

@bp.route('/', methods=['GET']) 
def home():
    if 'LABEL' not in session:
        session['LABEL'] = ''
        
    if 'img_grd_paths' not in session:
        session['img_grd_paths'] = []
        session['img_idx'] = []
        session['img_grd_c_labels'] = []
        session['C_LABEL'] = []
        session['C_LABELS'] = []
        session['NUM_IMGS'] = 0
        session['NUM_FILTERED'] = 0
        session['NUM_REFRESH'] = 0
        
    return render_template('/shuffle.html', 
            LABEL=session['LABEL'], NUM_IMGS=session['NUM_IMGS'], 
            NUM_FILTERED=session['NUM_FILTERED'], NUM_REFRESH=session['NUM_REFRESH'],
            C_LABELS=session['C_LABELS'], C_LABEL = session['C_LABEL'],
            imgs=zip(session['img_grd_paths'], session['img_idx'], session['img_grd_c_labels']))


@bp.route('/upload', methods=['POST']) 
def upload():
    if not os.path.exists(current_app.config['UPLOAD_DIR']):
        os.makedirs(current_app.config['UPLOAD_DIR'])

    if request.method == 'POST':
        z_file = request.files['file']
        if z_file and is_zipfile(z_file.filename):
            if not os.path.exists(current_app.config['UPLOAD_DIR']):
                os.makedirs(current_app.config['UPLOAD_DIR'])
                
            if request.method == 'POST':
                z_file = request.files['file']
                if z_file and is_zipfile(z_file.filename):
                    zfname = z_file.filename
                    zpath = os.path.join(current_app.config['UPLOAD_DIR'], zfname)
                    z_file.save(zpath)
                    session['zpath'] = zpath

        ## for multi file img upload
        # for key, f in request.files.items():  
        #     if key.startswith('file') and allowed_file(f.filename):
        #         f = secure_filename(f.filename)
        #         f.save(os.path.join(current_app.config['UPLOAD_DIR'], f.filename))

    return redirect('/')

    
@bp.route('/tasks/<task_type>', methods=['POST'])
def run_task(task_type):    # TODO: split into different routes
    task_data = {}
    if request.is_json:
        req = request.get_json()
        pprint.pprint(req)
        task_data = req['task_data']
    
    if task_type=='extract_zip':  
        zfname = os.path.basename(os.path.normpath(session['zpath']))
        task_data['progress_msg'] = 'Uploading <b>[ {} ]</b> ...'.format(zfname)
        task = current_app.task_queue.enqueue(extract_zip, session['zpath'], job_timeout=180)  

    if task_type=='load_data':
        upload_dir = os.path.join(current_app.config['UPLOAD_DIR'], session['LABEL'])
        task = current_app.task_queue.enqueue(load_data, (session['LABEL'], upload_dir))  

    if task_type=='train':
        print(type(current_app.config['OUTPUT_DIR']))
        dataset = ImageBucket(output_dir=current_app.config['OUTPUT_DIR'])
        
        task_data['NUM_IMGS'] = len(dataset)
        task_data['NUM_TRAIN'] = len(dataset.train)
        task_data['NUM_TEST'] = len(dataset.test)

        session['NUM_IMGS'] = len(dataset)
        session['NUM_TRAIN'] = len(dataset.train)
        session['NUM_TEST'] = len(dataset.test)
        
        print(dataset)
        task = current_app.task_queue.enqueue(train, dataset, job_timeout=600)  # 10 min training
        task_type = 'train'

    if task_type=='cluster': # Rerunning HDBSCAN
        task = current_app.task_queue.enqueue(cluster, session['LABEL'], job_timeout=180)
        session['NUM_CLUSTERS'] = 0
        session['C_LABELS'] = []
        
    if task_type=='som':    # Resetting the SOM
        # Resetting processed in db
        for img in Image.query.filter_by(label=session['LABEL']).filter_by(processed=True):
            img.reset()   

        ## Init vars
        if session['C_LABELS'] == []:
            c_labels = np.load(os.path.join(current_app.config['OUTPUT_DIR'], '_c_labels.npy'))
            session['C_LABELS'] = c_labels
        session['C_LABELS'] = [int(x)  for x in set(session['C_LABELS'])]
        task_data['C_LABELS'] = session['C_LABELS']
        session['img_grd_paths'] =[]
        session['img_idx'] =[]
        session['img_grd_c_labels']=[]

        # if 'C_LABEL' not in task_data: ## Running first time
        #     task_data['C_LABEL'] = session['C_LABELS'][0]   # 0
        #     session['C_LABEL'] = task_data['C_LABEL']

        #     task = run_new_som(task_data['C_LABEL'])

        print(task_data['C_LABEL'], session['C_LABEL'])
        ## Switching between different SOMS depending on task_data['C_LABEL']
        som_path = os.path.join(current_app.config['OUTPUT_DIR'], '_som_{}.npy'.format(task_data['C_LABEL']))
        if task_data['SOM_MODE'] == 'switch':
            if os.path.exists(som_path):
                imgs = Image.query.filter_by(label=session['LABEL']) \
                                .filter_by(processed=False).filter_by(c_label=int(task_data['C_LABEL'])).all()
                img_idx = [img.idx for img in imgs]
                
                task = current_app.task_queue.enqueue(som, (img_idx, task_data['C_LABEL'], 'switch'))
                session['C_LABEL'] = task_data['C_LABEL'] 
                session['NUM_IMGS']  = len(img_idx)

                # Saving NUM_FILTERED, NUM_REFRESH
                som_net = np.load(som_path).item()
                som_net['NUM_IMGS'] = session['NUM_IMGS']
                som_net['NUM_FILTERED'] = session['NUM_FILTERED']
                som_net['NUM_REFRESH'] = session['NUM_REFRESH']
                np.save(som_path, som_net)
            else:     # Load new SOM
                task = run_new_som(task_data['C_LABEL'])
                session['C_LABEL'] = task_data['C_LABEL']

        if task_data['SOM_MODE'] == 'new':   # If SOM button clicked
            ## Refreshing SOM with current task_data['C_LABEL']
            task_data['C_LABEL'] = session['C_LABEL']
            task = run_new_som(task_data['C_LABEL'])

    response_object = {
        'status': 'success',
        'task': {
            'task_type': task_type,
            'task_id': task.get_id(),
            'task_data': task_data    
        }
    }
    return jsonify(response_object), 202



@bp.route('/tasks/<task_type>/<task_id>', methods=['POST'])
def get_status(task_type, task_id):
    task = current_app.task_queue.fetch_job(task_id)
    if request.is_json:
        req = request.get_json()
        pprint.pprint(req)
        task_data = req['task_data']
        
    #TODO: create separate routes instead
    if task_type=='extract_zip' and task.get_status()=='finished':
        label = task.result
        session['LABEL'] = label
        task_data['LABEL'] = label
        
        upload_dir = os.path.join(current_app.config['UPLOAD_DIR'], session['LABEL'])
        task = current_app.task_queue.enqueue(load_data, (session['LABEL'], upload_dir))
        task_type = 'load_data'
        
    elif task_type=='extract_zip':    # Report job progress
        task_data.update(task.meta)
        task.refresh()

    elif task_type=='load_data' and task.get_status()=='finished':
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
        print(session['LABEL'])
        task = current_app.task_queue.enqueue(cluster, session['LABEL'], job_timeout=300)
        
        session['NUM_CLUSTERS'] = 0
        session['C_LABELS'] = []
        task_type = 'cluster'
        
    elif task_type=='train':    # Report job progress
        task_data.update(task.meta)
        task.refresh()

    elif task_type=='cluster' and task.get_status()=='finished':
        feat, c_labels, imgs = task.result
        task_data['NUM_CLUSTERS'] = len(set(c_labels))
        session['NUM_CLUSTERS'] = len(set(c_labels))   
        
        save_img_db(c_labels)

        session['img_grd_paths'] =[]
        session['img_idx'] =[]
        session['img_grd_c_labels']=[]
        session['C_LABELS'] = [int(x)  for x in set(c_labels)]
        session['C_LABEL'] = session['C_LABELS'][0]
        
        task = run_new_som(session['C_LABEL'])

        task_data['C_LABEL'] = session['C_LABEL']
        task_data['C_LABELS'] = session['C_LABELS'] 
        task_data['SOM_MODE'] = 'new'
        task_type = 'som'

    elif task_type=='cluster':
        task_data.update(task.meta)
        task.refresh()
            
    elif task_type=='som' and task.get_status()=='finished':
        img_grd_idx, c_label = task.result
        print(img_grd_idx)
        img_grd = ImageGrid(img_grd_idx)    # Update img_grd
        
        session['img_grd_paths'] = img_grd.img_paths
        session['img_idx']= img_grd.img_idx
        session['img_grd_c_labels'] = img_grd.c_labels

        task_data['NUM_IMGS'] = session['NUM_IMGS']
        task_data['NUM_FILTERED'] =  session['NUM_FILTERED']
        task_data['NUM_REFRESH'] = session['NUM_REFRESH'] 

        # Loading NUM_IMGS, NUM_FILTERED, NUM_REFRESH from
        if task_data['SOM_MODE'] == 'switch':
            som_path = os.path.join(current_app.config['OUTPUT_DIR'], '_som_{}.npy'.format(task_data['C_LABEL']))
            som_net = np.load(som_path).item()
            if 'NUM_IMGS' in som_net:
                session['NUM_IMGS'] = som_net['NUM_IMGS']
                session['NUM_FILTERED'] = som_net['NUM_FILTERED']
                session['NUM_REFRESH'] = som_net['NUM_REFRESH']

        ##  Trying to load all img grid paths at once
        # session['img_grd_paths'].append(img_grd.img_paths)
        # session['img_idx'].append(img_grd.img_idx)
        # session['img_grd_c_labels'].append(img_grd.c_labels)
        # if task_data['C_LABEL'] != task_data['C_LABELS'][-1]:
        #     print(task_data['C_LABELS'][task_data['C_LABEL']+1])
        #     task_data['C_LABEL'] = task_data['C_LABELS'][task_data['C_LABEL']+1]
        #     session['C_LABEL'] = task_data['C_LABEL']
        #     task = run_new_som(task_data['C_LABEL'])
        # else:
        #     session['img_grd_paths'] = task_data['img_grd_paths']
        #     session['img_idx'] =  task_data['img_idx']
        #     session['img_grd_c_labels'] = task_data['img_grd_c_labels']

        #     print(session['img_grd_paths'], len(session['img_grd_paths']))

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

    if request.is_json:
        req = request.get_json()
        pprint.pprint(req)
        task_data = req['task_data']
    
    # Setting all imgs in grd to processed
    seen = [Image.query.filter_by(label=session['LABEL'])
                .filter_by(idx=int(idx)).first().seen() for idx in img_idx]
    selected = [Image.query.filter_by(label=session['LABEL'])
                .filter_by(idx=int(idx)).first() for idx in selected_img_idx]
    
    # Get new unprocessed imgs to replace img_grd_idx 
    imgs = Image.query.filter_by(label=session['LABEL']) \
                        .filter_by(processed=False).filter_by(c_label=int(session['C_LABEL'])).all()
    img_idx = [img.idx for img in imgs]

    # Update SOM
    task = current_app.task_queue.enqueue(som, (img_idx, session['C_LABEL'], 'update'))

    # Updating num imgs seen
    session['NUM_IMGS'] = len(imgs)
    session['NUM_FILTERED'] += len(selected)
    session['NUM_REFRESH'] += 1
    
    task_data.update({'LABEL': session['LABEL'],
                    'C_LABEL': session['C_LABEL'],
                    'NUM_REFRESH': session['NUM_REFRESH'], 
                    'NUM_FILTERED': session['NUM_FILTERED'],
                    'NUM_IMGS': session['NUM_IMGS']
                    })
        
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
            'NUM_IMGS': session['NUM_IMGS'],
            'NUM_REFRESH': session['NUM_REFRESH']
        }
    }
    return jsonify(response_object), 202

def run_new_som(c_label):
    if 'output_dir' not in session: 
        session['output_dir'] = current_app.config['OUTPUT_DIR']
        
    if os.path.exists(os.path.join(session['output_dir'], '_som_{}.npy'.format(c_label))): # Clearing som weights
        os.remove(os.path.join(session['output_dir'], '_som_{}.npy'.format(c_label)))  
        
    img_idx = [img.idx for img in Image.query.filter_by(label=session['LABEL'])
                                        .filter_by(c_label=int(c_label)).all()]

    task = current_app.task_queue.enqueue(som, (img_idx, c_label, 'new'), job_timeout=180)

    # Resetting session vars
    session['NUM_REFRESH'] = 0
    session['NUM_FILTERED'] = 0 
    session['NUM_IMGS'] = len(img_idx)

    return task



# TODO: Should be done as tasks but yet to work out how to declare db instance in tasks.py ><
def save_img_db(c_labels):
    img_names = os.listdir(current_app.config['IMG_DIR'])
    img_paths = [url_for('static', filename=os.path.join('imgs', img)) for img in img_names]
    clear_tables()
    for img_path in img_paths:
        idx = int(os.path.basename(os.path.normpath(img_path)).split('.')[0])
        img = Image(idx=idx, label=session['LABEL'], c_label=int(c_labels[idx]), 
                img_path=img_path, processed=False).add()
        print(idx, img_path, c_labels[idx])