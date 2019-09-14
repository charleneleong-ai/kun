#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, September 5th 2019, 9:14:26 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sat Sep 14 2019
###

import warnings
warnings.filterwarnings('ignore')

import glob
import os
from datetime import datetime

from flask import Flask
from flask import render_template
from flask import request, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy

from model.ae import AutoEncoder

# TODO: remove later
import sys
sys.path.append('../models/')
from utils.datasets import FilteredMNIST
from utils.plt import plt_scatter

ROOT_DIR = os.path.dirname(__file__)
DB_FILE = 'sqlite:///{}'.format(os.path.join(ROOT_DIR, 'images.db'))

HOST = '0.0.0.0'
PORT = 8888

# OUTPUT_DIR = max(glob.iglob('./../models/output/*output/'), key=os.path.getctime)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_FILE
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Image(db.Model):
    idx = db.Column(db.Integer, primary_key=True, nullable=False)
    label = db.Column(db.Integer, nullable=False)
    image_path = db.Column(db.String, nullable=False)


@app.route('/', methods=['GET', 'POST'])
def home():
    images = glob.glob('static/imgs/*.png')
    return render_template('home.html', images=images)

@app.route('/upload', methods=['POST'])
def upload():
    dataset = FilteredMNIST(label=8, split=0.8, n_noise_clusters=3)
    train(dataset)
    return redirect('/')

@app.route('/train', methods=['POST'])
def train(dataset):
    EPOCHS = 50
    BATCH_SIZE = 128
    LR = 1e-3       
    N_TEST_IMGS = 8

    ae = AutoEncoder()

    timestamp = datetime.now().strftime('%Y.%d.%m-%H:%M:%S')
    OUTPUT_DIR = './output/{}_{}_{}'.format('ae', dataset.LABEL, timestamp)
    print(OUTPUT_DIR)
    ae.fit(dataset, 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            lr=LR, 
            opt='Adam',         # Adam
            loss='BCE',         # BCE or MSE
            patience=10,        # Num epochs for early stopping
            eval=True,          # Eval training process with test data
            plt_imgs=(N_TEST_IMGS, 10),         # (N_TEST_IMGS, plt_interval)
            scatter_plt=('tsne', 10),           # ('method', plt_interval)
            output_dir=OUTPUT_DIR, 
            save_model=True)        # Also saves dataset
    
    return redirect('/')

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

if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)
