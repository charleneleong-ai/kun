#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, September 5th 2019, 9:14:26 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sun Sep 15 2019
###

import warnings
warnings.filterwarnings('ignore')

import glob
import os
from datetime import datetime

from flask import Flask
from flask import render_template, url_for
from flask import request, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy

from server import create_app

import redis
from rq import Connection, Worker

from server.tasks import train, filtered_MNIST


HOST = '0.0.0.0'
PORT = 8888

# OUTPUT_DIR = max(glob.iglob('./../models/output/*output/'), key=os.path.getctime)

# app = Flask(__name__,
#         template_folder='./client/templates',
#         static_folder='./client/static'
#         )
        

# r = redis.Redis()
# q = Queue(connection=r)

# db = SQLAlchemy(app)

# app = create_app()

# db = SQLAlchemy(app)

app = create_app()


# @cli.command('run_worker')
def run_worker():
    redis_url = app.config['REDIS_URL']
    redis_connection = redis.from_url(redis_url)
    with Connection(redis_connection):
        worker = Worker(app.config['QUEUES'])
        worker.work()

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     images = os.listdir(os.path.join(app.static_folder, 'imgs'))
#     images = [url_for('static', filename=os.path.join('imgs', img)) for img in images]
#     return render_template('home.html', images=images)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     return render_template('home.html')
    
# @app.route('/upload', methods=['POST'])
# def upload():
#     label = 8
#     task = q.enqueue(filtered_MNIST, 8)  # Send a training job to the task queue

#     response = {
#         'status': 'success',
#         'data': {
#             'task_id': task.get_id()
#         }
#     }

#     return render_template('home.html', response=response)

# # @app.route('/train', methods=['POST'])
# # tasks

# @app.route('/upload/<task_id>', methods=['GET'])
# def get_status(task_id):
#     # with Connection(redis.from_url(current_app.config['REDIS_URL'])):
#     #     q = Queue()
#     task = q.fetch_job(task_id)
#     if task:
#         response_object = {
#             'status': 'success',
#             'data': {
#                 'task_id': task.get_id(),
#                 'task_status': task.get_status(),
#                 'task_result': task.result,
#             }
#         }
#     else:
#         response_object = {'status': 'error'}
#     return jsonify(response_object)


def add_task():
    jobs = q.jobs  # Get a list of jobs in the queue
    message = None

    if request.args:  # Only run if a query string is sent in the request

        url = request.args.get("url")  # Gets the URL coming in as a query string

        task = q.enqueue(count_words, url)  # Send a job to the task queue

        jobs = q.jobs  # Get a list of jobs in the queue

        q_len = len(q)  # Get the queue length

        message = f"Task queued at {task.enqueued_at.strftime('%a, %d %b %Y %H:%M:%S')}. {q_len} jobs queued"

    return render_template("add_task.html", message=message, jobs=jobs)

def run_task():
    task_type = request.form['type']
    with Connection(redis.from_url(app.config['REDIS_URL'])):
        q = Queue()
        task = q.enqueue(create_task, task_type)
    response_object = {
        'status': 'success',
        'data': {
            'task_id': task.get_id()
        }
    }
    return jsonify(response_object), 202

def load_model(dataset):
    # assume latest model
    OUTPUT_DIR = max(glob.iglob('./output/*/'), key=os.path.getctime)
    _, feat, labels, imgs = ae.eval_model(dataset=dataset, output_dir=OUTPUT_DIR)


# def preprocess(): 
#     transforms.Compose([
#         transforms.Resize(28),
#         transforms.ToTensor()])

# def predict(img):
#     preprocess = transforms.Compose([
#                            transforms.Resize(28),
#                            transforms.ToTensor(),
#                         ])
#     ae.eval()
#     with torch.no_grad():
#         # img_tensor = self.preprocess(img) # tensor in [0,1]
#         img_tensor = preprocess(img)
#         img_tensor = 1 - img_tensor
#         img_tensor = img_tensor.view(1, 28, 28, 1).to(ae.device)

#         # Do Inference
#         encoded, decoded = ae.forward(img_tensor.view(-1, 784).to(ae.device))

#         # probabilities = self.net(img_tensor)
#         encoded
#         probabilities = F.softmax(probabilities, dim = 1)

#     return probabilities[0].cpu().numpy()

# @app.route('/predict', methods=['GET','POST'])
# def predict():
#     results = {'prediction' :'Empty', 'probability' :{}}

    # get data
#     input_img = BytesIO(base64.urlsafe_b64decode(request.form['img']))
# #   Open the Image and resize
#     input_img = Image.open(input_img).convert('L')

    # model.predict method takes the raw data and output a vector of probabilities
    # res =  ae.predict(input_img)


    # results['prediction'] = str(CLASS_MAPPING[np.argmax(res)])
    # results['probability'] = float(np.max(res))*100
    # results['prediction'] = 5
    # results['probability'] = 50.424

    # output data
    # return json.dumps(results)


def run_worker():
    redis_url = app.config['REDIS_URL']
    redis_connection = redis.from_url(redis_url)
    with Connection(redis_connection):
        worker = Worker(app.config['QUEUES'])
        worker.work()


if __name__ == '__main__':
    # run web server
    # run_worker()
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)
