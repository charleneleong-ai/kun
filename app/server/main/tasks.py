#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 4:18:39 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Thu Oct 03 2019
###
import os
import shutil
import glob
from datetime import datetime
import json, codecs
from rq import get_current_job

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min
from hdbscan import HDBSCAN
from torchvision.utils import save_image, make_grid

from server.__init__ import create_app
from server.model.ae import AutoEncoder
from server.model.som import SOM
from server.model.utils.plt import plt_scatter
from server.utils.datasets.filteredMNIST import FilteredMNIST
from server.utils.datasets.imgbucket import ImageBucket
from server.main.models import Image
from server.utils.load import np_json, json_np

# Import current app settings for app config
app = create_app()
app.app_context().push()


SEED = 489

def upload_MNIST(label):
    return FilteredMNIST(label=label, split=0.8, n_noise_clusters=3, download_dir=app.config['RAW_IMG_DIR'])
    
def upload(args):
    label = args[0]
    img_dir = args[1]
    return ImageBucket(label=str(label), split=0.8, img_dir=img_dir, download_raw=False, download_dir=app.config['RAW_IMG_DIR'])


def train(dataset):
    job = get_current_job()
    if len(dataset.train) < 500:
        BATCH_SIZE = 32
    elif len(dataset.train) < 2000:
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 128
    
    MAX_EPOCHS = 50
    LR = 1e-3       
    N_TEST_IMGS = 8
    PATIENCE = 10
    
    job.meta['BS'] = BATCH_SIZE
    job.meta['MAX_EPOCHS'] = MAX_EPOCHS
    job.meta['LR'] = LR
    job.meta['PATIENCE'] = PATIENCE
    job.save_meta()

    ae = AutoEncoder(job=job)
    timestamp = datetime.now().strftime('%Y.%m.%d-%H%M%S')
    MODEL_OUTPUT_DIR = app.config['MODEL_OUTPUT_DIR']
    OUTPUT_DIR = os.path.join(MODEL_OUTPUT_DIR, '{}_{}_{}'.format('ae', dataset.LABEL, timestamp))
    print(OUTPUT_DIR)
    ae.fit(dataset, 
            batch_size=BATCH_SIZE, 
            max_epochs=MAX_EPOCHS, 
            lr=LR, 
            opt='Adam',         # Adam
            loss='BCE',         # BCE or MSE
            patience=PATIENCE,        # Num epochs for early stopping
            eval=True,          # Eval training process with test data
            # plt_imgs=(N_TEST_IMGS, 10),         # (N_TEST_IMGS, plt_interval)
            scatter_plt=('tsne', 10),           # ('method', plt_interval)
            output_dir=OUTPUT_DIR, 
            save_model=True)        # Also saves dataset
    
    return BATCH_SIZE, LR, ae.EPOCH, OUTPUT_DIR

def cluster(label):
    # Returns the  OUTPUT DIR of the latest model by default
    OUTPUT_DIR = app.config['OUTPUT_DIR']
    ae, feat_ae, labels, imgs = load_model(OUTPUT_DIR)
    clear_cluster_output(OUTPUT_DIR)   # Clearing old output
    
    print('Clustering' , label, 'with HDBSCAN...\n')
    feat = tsne(feat_ae, 2)     # HDBSCAN works best with 2dim w/ small dataset 
    c_labels = hdbscan(feat, min_cluster_size=10)   
    c_labels = sort_c_labels(c_labels)
    
    img_plt = plt_scatter(feat, c_labels, output_dir=OUTPUT_DIR, plt_name='_{}.png'.format('hdbscan'), pltshow=False)
    ae.tb.add_image(tag='_{}'.format('hdbscan'), 
                    img_tensor=img_plt, 
                    global_step = ae.EPOCH, dataformats='HWC')

    print('Saving images to client/static/imgs...')
    shutil.rmtree(app.config['IMG_DIR'])    # Clear img dir
    os.makedirs(app.config['IMG_DIR'])
    for i, img in enumerate(imgs):
        save_image(img.view(-1, 1, 28, 28), app.config['IMG_DIR']+'/{}.png'.format(i))

    print('Saving processed ae feat...')      # Saving feat to json bcz tsne slow
    np_json(feat, os.path.join(OUTPUT_DIR, '_feat.json'))

    return feat, c_labels

def som(args):
    img_idx = np.array(args[0])
    c_labels = np.array(args[1])
    job = get_current_job()

    # returns the  OUTPUT DIR of the latest model by default
    OUTPUT_DIR = app.config['OUTPUT_DIR']
    ae, feat_ae, labels, imgs = load_model(OUTPUT_DIR)  
    feat = json_np(os.path.join(OUTPUT_DIR, '_feat.json'))
    lut = dict(enumerate(list(img_idx)))

    data = feat[img_idx]

    dims= [10, 20]  # dims[row, col]
    som_path = os.path.join(OUTPUT_DIR, '_som.json')
    if os.path.exists(som_path):  # Declare new SOM else update net
        iter = 50
        lr = 0.0001 
        som = SOM(data=data, dims=dims, n_iter = iter, lr_init=lr, net_path=som_path, job=job)
    else:
        iter = 3000
        lr = 0.2  
        som = SOM(data=data, dims=dims, n_iter = iter, lr_init=lr, job=job)
    
    job.meta['MAX_ITER'] = iter
    job.meta['LR'] = lr
    job.meta['DIMS'] = dims
    job.save_meta()

    print('Ordering image grid with Self Organising Map...')
    print('iter: {} lr: {}\n'.format(iter, lr))
    net = som.train()


    net_w = np.array([net[x, y, :] for x in range(net.shape[0]) for y in range(net.shape[1])])
    img_grd_idx, _ = pairwise_distances_argmin_min(net_w, data)
    img_grd_idx = np.array([lut[i] for i in img_grd_idx])   # Remapping to img_idx indices

    # Plotting HDBSCAN SOM grid 
    img_plt = plt_scatter([feat, feat[img_grd_idx]], c_labels, colors=['blue'], 
                            output_dir=OUTPUT_DIR, 
                            plt_name='_{}_som_2D_{}_lr={}.png'.format('hdbscan', iter, lr), 
                            pltshow=False, 
                            plt_grd_dims=dims)
    ae.tb.add_image(tag='_{}_som_2D_{}_lr={}.png'.format('hdbscan', iter, lr), 
                                    img_tensor=img_plt, 
                                    global_step = ae.EPOCH, dataformats='HWC')

    if not os.path.exists(som_path):                            
        # Saving image grid
        img_grd = imgs[img_grd_idx].view(-1, 1, 28, 28)
        save_image(img_grd, OUTPUT_DIR+'/_img_grd_som_2D_{}_lr={}.png'.format(iter, lr), nrow=20)    # nrow = num in row
        ae.tb.add_image(tag='_img_grd_som_2D_{}_lr={}.png'.format(iter, lr), 
                                img_tensor=make_grid(img_grd, nrow=20), 
                                global_step = ae.EPOCH)
    
    print('Saving SOM weights...')
    np_json(net, som_path)
                            
    return img_grd_idx

                            
# Helper functions for cluster()
def load_model(output_dir):
    ae = AutoEncoder()  
    # dataset = FilteredMNIST(output_dir=output_dir)
    dataset = ImageBucket(output_dir=output_dir)
    model_path = ae.load_model(output_dir=output_dir)
    dataset.test += dataset.train   # Get all the data, eval_model loads dataset.test
    _, feat, labels, imgs = ae.eval_model(dataset=dataset, output_dir=output_dir)
    return ae, feat, labels, imgs
    
def tsne(feat, dim):
    tsne = TSNE(perplexity=30, n_components=dim, init='pca', n_iter=1000, random_state=SEED)
    feat = tsne.fit_transform(feat)
    feat = MinMaxScaler().fit_transform(feat) 
    return feat

def hdbscan(feat, min_cluster_size):
    cluster = HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=False)
    cluster.fit(feat)
    return cluster.labels_

def sort_c_labels(c_labels):
    cluster_size = [(label, len(c_labels[c_labels==label])) for label in np.unique(c_labels[c_labels!=-1])] # Tuple
    sorted_cluster_size = sorted(cluster_size, key=lambda x:x[1])[::-1] # Sort by cluster_size[1] large to small
    sorted_cluster_size = [c[0] for c in sorted_cluster_size]   # Return idx
    lut = np.array([i[0] for i in sorted(enumerate(sorted_cluster_size), key=lambda x:x[1])])
    c_labels[c_labels!=-1] = lut[c_labels[c_labels!=-1]]    # Keep noise c_labels
    return c_labels

def clear_cluster_output(output_dir):
    files = glob.glob(output_dir+'_*')
    for f in files:
        os.remove(f)
