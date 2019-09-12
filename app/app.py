#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, September 5th 2019, 9:14:26 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Thu Sep 12 2019
###
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../models/')
import glob
import os

from flask import Flask, request, jsonify, render_template, render_template_string

from ae.ae import AutoEncoder
from ae.convae import ConvAutoEncoder
from utils.datasets import FilteredMNIST
from utils.plt import plt_scatter


HOST = '0.0.0.0'
PORT = 8888

OUTPUT_DIR = max(glob.iglob('./../models/output/*output/'), key=os.path.getctime)
print(OUTPUT_DIR)

app = Flask(__name__)


files = glob.glob('./imgs/*.png')
print(files)

# ae = AutoEncoder().load_model(output_dir=OUTPUT_DIR)


@app.route('/')
def home():
    images = glob.glob('static/imgs/*.png')
    return render_template('home.html', images=images)


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
