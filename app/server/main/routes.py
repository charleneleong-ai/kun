#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 6:35:09 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sun Oct 06 2019
###

# app/server/main/routes.py

import os
import glob
import shutil
from zipfile import ZipFile, BadZipfile
import chardet
from copy import copy

from flask import render_template, Blueprint
from flask import jsonify, url_for, request, redirect, flash
from flask import current_app, session

from server.main import bp
from server.main.tasks import train, load_data, load_MNIST, cluster, som
from server.main.models import Image, ImageGrid, clear_tables
from server.utils.load import zh_detect

import pprint

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def is_zipfile(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in set(['zip'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/', methods=['GET']) 
def home():
    if 'LABEL' not in session:
        session['LABEL'] = ''
        
    if 'img_grd_paths' not in session:
        session['img_grd_paths'] = []
        session['img_idx'] = []
        session['c_labels'] = []
        session['img_grd_c_labels'] = []
        session['NUM_SEEN'] = 0
        session['NUM_FILTERED'] = 0
        session['NUM_CLUSTERS'] = 0
        session['NUM_REFRESH'] = 0
        
    return render_template('/shuffle.html', 
            LABEL=session['LABEL'], 
            NUM_SEEN=session['NUM_SEEN'], NUM_FILTERED=session['NUM_FILTERED'], 
            NUM_CLUSTERS=session['NUM_CLUSTERS'], NUM_REFRESH=session['NUM_REFRESH'],
            img_grd_c_labels=session['img_grd_c_labels'], 
            imgs=zip(session['img_grd_paths'], session['img_idx'], session['c_labels']))


@bp.route('/upload', methods=['POST']) 
def upload():
    if not os.path.exists(current_app.config['UPLOAD_DIR']):
        os.makedirs(current_app.config['UPLOAD_DIR'])
        
    if request.method == 'POST':
        ## for multi file upload
        # for key, f in request.files.items():  
        #     if key.startswith('file') and allowed_file(f.filename):
        #         f = secure_filename(f.filename)
        #         f.save(os.path.join(current_app.config['UPLOAD_DIR'], f.filename))
        z_file = request.files['file']
        if z_file and is_zipfile(z_file.filename):
            zfname = z_file.filename
            zpath = os.path.join(current_app.config['UPLOAD_DIR'], zfname)
            char = zfname.split('.')[0]
            z_file.save(zpath)
            zf = test_zipfile(zpath)
            
            print(char)
            # img_dir = os.path.join(current_app.config['UPLOAD_DIR'], zfname.split('.')[0])
            # if os.path.exists(os.path.join(current_app.config['DATASET_DIR'], zfname.split('.')[0])):
            #     flash('This label has already been uploaded') 
            #     redirect(request.url)

            ## ZIP files are encoded as in CP437, if they cannot decide the original encoding.
            ## So must decode for zh chars
            # https://leeifrankjaw.github.io/articles/fix_filename_encoding_for_zip_archive_with_python.html
            if zh_detect(char):
                zfname = char+'_decoded.zip'
                with ZipFile(os.path.join(current_app.config['UPLOAD_DIR'], zfname), mode='w') as ztf:
                    ztf.comment = zf.comment
                    for zinfo in zf.infolist():
                        zinfo.CRC = None
                        ztinfo = copy(zinfo)
                        ztinfo.filename = zinfo.filename.encode('cp437').decode('utf8')
                        ztf.writestr(ztinfo, zf.read(zinfo))
                zf.close()
                print( ztf.namelist())

            print(os.path.join(current_app.config['UPLOAD_DIR'], zfname))
                
            with ZipFile(os.path.join(current_app.config['UPLOAD_DIR'], zfname),"r") as z_ref:
                print('Extracting')
                z_ref.extractall(current_app.config['UPLOAD_DIR'])
            z_ref.close()
            
            session['LABEL'] = char
            
    return redirect('/')

    
@bp.route('/tasks/<task_type>', methods=['POST'])
def run_task(task_type):
    task_data = {'LABEL':  session['LABEL'] }
    
    if task_type=='load_data':
        upload_dir = os.path.join(current_app.config['UPLOAD_DIR'], session['LABEL'])
        task = current_app.task_queue.enqueue(load_data, (session['LABEL'], upload_dir))  
        
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
        if os.path.exists(os.path.join(session['output_dir'], '_som.json')): # Clearing som weights
            os.remove(os.path.join(session['output_dir'], '_som.json'))  
        
        task = run_new_som()
        
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
        print(req)
        task_data = req['task_data']
        
    #TODO: create separate routes instead
    if task_type=='load_data' and task.get_status()=='finished':
        clear_upload_folder()
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
        feat, c_labels, imgs = task.result
        task_data['NUM_CLUSTERS'] = len(set(c_labels))
        session['NUM_CLUSTERS'] = len(set(c_labels))   
        print(len(c_labels), len(feat))
        
        # print('Saving images to client/static/imgs...')
        # if os.path.exists(current_app.config['IMG_DIR']): # Clearing img dir
        #     shutil.rmtree(current_app.config['IMG_DIR'])    
        # os.makedirs(current_app.config['IMG_DIR'])
        # clear_tables()
        # for idx, (c_label, img) in zip(c_labels, imgs):
        #     img_path = current_app.config['IMG_DIR']+'/{}.png'.format(idx)
        #     save_image(img.view(-1, 1, 28, 28), img_path)
        #     img = Image(idx=idx, label=session['LABEL'], c_label=int(c_label), 
        #                 img_path=img_path, processed=False).add()

        save_img_db(c_labels)
                
        task = run_new_som()
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
    # seen = [Image.query.filter_by(label=session['LABEL'])
    #             .filter_by(idx=int(idx)).first().seen() for idx in img_idx]
    # selected = [Image.query.filter_by(label=session['LABEL'])
    #             .filter_by(idx=int(idx)).first() for idx in selected_img_idx]
    
    # # Get new unprocessed imgs to replace img_grd_idx 
    # imgs = Image.query.filter_by(label=session['LABEL']) \
    #                     .filter_by(processed=False).order_by(Image.c_label.desc()).all()
    # img_idx = [img.idx for img in imgs]
    # c_labels = [img.c_label for img in Image.query.filter_by(label=session['LABEL']).all()]

    seen = [Image.query.filter_by(idx=int(idx)).first().seen() for idx in img_idx]
    selected = [Image.query.filter_by(idx=int(idx)).first() for idx in selected_img_idx]
    
    # Get new imgs to replace seen img_grd_idx
    imgs = Image.query.filter_by(processed=False).order_by(Image.c_label.desc()).all()
    img_idx = [img.idx for img in imgs]

    task = current_app.task_queue.enqueue(som, img_idx, job_timeout=300)

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

def run_new_som():
    img_idx = [img.idx for img in Image.query.filter_by(label=session['LABEL'])
                                        .order_by(Image.c_label.desc()).all()]

    task = current_app.task_queue.enqueue(som, img_idx, job_timeout=180)

    # Resetting session vars
    session['NUM_REFRESH'] = 0
    session['NUM_FILTERED'] = 0 
    session['NUM_SEEN'] = len(img_idx)

    return task

def test_zipfile(zfname):
    try:
        zf = ZipFile(zfname)
    except BadZipfile as e:
        raise IOError(e)
    except RuntimeError as e:
        if "encrypted" in e.args[0] or "Bad password" in e.args[0]:
            raise PasswordError(e)
        else:
            raise CRCError(e)
    except Exception as e:
        raise IOError(e)
    return zf


def clear_upload_folder():
    if os.path.exists(os.path.join(current_app.config['DATASET_DIR'], session['LABEL'])):
        shutil.rmtree(current_app.config['UPLOAD_DIR'])


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