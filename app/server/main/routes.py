#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 6:35:09 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Sep 20 2019
###

# project/server/main/views.py

import os

import redis
from rq import Queue, Connection
from flask import render_template, Blueprint, jsonify, \
    url_for, request, redirect, current_app, session
    
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from server.worker import conn
from server.main import bp
from server.main.tasks import train, filtered_MNIST, cluster, som
from server.main.models import Image, ImageGrid, clear_tables



@bp.route('/', methods=['GET'])
def home():
    # img_names = os.listdir(current_app.config['IMG_GRD_DIR'])
    # imgs = [url_for('static', filename=os.path.join('img_grd', img)) for img in img_names]
    if 'img_grd_paths' not in session:
        session['img_grd_paths'] = []
        session['img_idx'] = []
 
    return render_template('/shuffle.html', imgs=zip(session['img_grd_paths'], session['img_idx']))


@bp.route('/tasks/<task_type>', methods=['POST'])
def run_task(task_type):
    if task_type=='upload':
        task = current_app.task_queue.enqueue(filtered_MNIST, 8)
    if task_type=='cluster':
        task = current_app.task_queue.enqueue(cluster, job_timeout=180)
       
    if task_type=='save_imgs':
        label_0_db = Image.query.filter_by(c_label=0).all()
        label_0_idx = [img.idx for img in label_0_db]
        
        task = current_app.task_queue.enqueue(som, label_0_idx, job_timeout=180)
        # temp will load from cluster task later

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
    #TODO: create separate routes instead
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
    elif task_type=='save_imgs' and task.get_status()=='finished':
        task_type = 'som'
        img_grd_idx = task.result
        img_grd = ImageGrid(img_grd_idx).imgs
        img_grd_paths = [img.img_path for img in img_grd]
        img_idx = [img.idx for img in img_grd]
        session['img_grd_paths'] = img_grd_paths
        print(img_idx)
        session['img_idx'] = img_idx


        # for img in img_grd_paths:
        #     plt.imread(img)
        #     save_image(img, app.config['IMG_GRD_DIR']+'/{}.png'.format(i)
        
        # for img in img_grd:
        #     print(img)
        
        # for i, label_0_idx in enumerate(img_grd_idx):
        #     print(i, label_0_idx, type(label_0_idx))
        #     img = Image.query.filter_by(idx=int(label_0_idx)).first()
        #     img.img_grd_idx = i
        #     Image.commit()

        # save_image(img.view(-1, 1, 28, 28), 
        #             current_app.config['IMG_GRD_DIR']+'/{}_{}.png'.format(i, label_0_idx))
        # print(img)
        
        # img_grd = Image.query.order_by(Image.img_grd_idx).filter(Image.img_grd_idx!=None).all()
        # print(len(img_grd))

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
        img = Image(idx=idx, c_label=int(c_labels[idx]), img_path=img_path, seen=False).add()
