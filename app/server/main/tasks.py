#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 4:18:39 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Fri Oct 18 2019
###

import warnings
warnings.filterwarnings('ignore')

import os
import shutil
import glob
from zipfile import ZipFile, BadZipfile
from copy import copy
from datetime import datetime
import json
from rq import get_current_job


import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min
from hdbscan import HDBSCAN
from umap import UMAP
from torchvision.utils import save_image, make_grid

from server.__init__ import create_app
from server.model.ae import AutoEncoder
from server.model.som import SOM
from server.model.utils.plt import plt_scatter, plt_scatter_3D
from server.utils.datasets.filteredMNIST import FilteredMNIST
from server.utils.datasets.imgbucket import ImageBucket
from server.main.models import Image
from server.utils.load import zh_detect

# Import current app settings for app config
app = create_app()
app.app_context().push()


SEED = 489
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def extract_zip(zpath):
    job = get_current_job()
    zf = test_zipfile(zpath)
    zfname = os.path.basename(os.path.normpath(zpath))
    label = zfname.split('.')[0]

    ## ZIP files are encoded as in CP437, if they cannot decide the original encoding.
    ## So must decode for zh chars
    # https://leeifrankjaw.github.io/articles/fix_filename_encoding_for_zip_archive_with_python.html
    if zh_detect(label):
        zfname = label+'_utf8.zip'
        job.meta['progress_msg'] = 'Fixing filename encoding ...'
        job.save_meta()
        with ZipFile(os.path.join(app.config['UPLOAD_DIR'], zfname), mode='w') as ztf:
            ztf.comment = zf.comment
            for zinfo in zf.infolist():
                zinfo.CRC = None
                ztinfo = copy(zinfo)
                ztinfo.filename = zinfo.filename.encode('cp437').decode('utf8')
                ztf.writestr(ztinfo, zf.read(zinfo))
        zf.close()
        print( ztf.namelist())

    print(os.path.join(app.config['UPLOAD_DIR'], zfname))
        
    with ZipFile(os.path.join(app.config['UPLOAD_DIR'], zfname),"r") as z_ref:
        job.meta['progress_msg'] = 'Extracting <b>[ {} ]</b> ...'.format(zfname)
        job.save_meta()
        print('Extracting {} ...'.format(zfname))
        z_ref.extractall(app.config['UPLOAD_DIR'])
    z_ref.close()
    
    return label
    

def load_MNIST(label):
    return FilteredMNIST(label=label, split=0.8, n_noise_clusters=3, download_dir=app.config['DATASET_DIR'])
    
    
def load_data(label, img_dir):
    clear_upload_folder(label)
    return ImageBucket(label=str(label), split=0.8, img_dir=img_dir, 
                        download_raw=False, download_dir=app.config['DATASET_DIR'])


def train(dataset):
    job = get_current_job()
    if len(dataset.train) < 500:
        BATCH_SIZE = 32
    elif len(dataset.train) < 2000:
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 128
    
    MAX_EPOCHS = 50
    LR = 0.001    
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
            scatter_plt=('umap', 10),           # ('method', plt_interval)
            output_dir=OUTPUT_DIR, 
            save_model=True)        # Also saves dataset
    
    return BATCH_SIZE, LR, ae.EPOCH, OUTPUT_DIR



def cluster(label):
    job = get_current_job()
   
    OUTPUT_DIR = app.config['OUTPUT_DIR']  # Returns the  OUTPUT DIR of the latest model by default
    ae, feat_ae, labels, imgs = load_model(OUTPUT_DIR)
    clear_output(OUTPUT_DIR)   # Clearing old output
    
    MIN_CLUSTER_SIZE = 15
    if len(feat_ae) > 5000:
        MIN_CLUSTER_SIZE = 15
    dim_reduce = 2
    dim_reduce_method = 'umap'  ## HDBSCAN suffers from curse of dimensionality 

    job.meta['MIN_CLUSTER_SIZE'] = MIN_CLUSTER_SIZE
    job.meta['progress_msg'] = 'Reducing features from {} to {} dims with {} ...'\
                                .format(feat_ae.shape[1], dim_reduce, dim_reduce_method.upper())
    job.save_meta()
   
    print('Reducing feat from {} to {} dims with {} ...'
            .format(feat_ae.shape[1], dim_reduce, dim_reduce_method.upper()))
    import time
    start = time.time()
    if dim_reduce_method=='tsne':
        feat = tsne(feat_ae, dim_reduce)  
        end = time.time()
    elif   dim_reduce_method=='umap':
        feat = umap(feat_ae, dim_reduce, MIN_CLUSTER_SIZE)
        end = time.time()
    print(end-start)
    print('Clustering' , label, 'with HDBSCAN ...')
    c_labels = hdbscan(feat, min_cluster_size=MIN_CLUSTER_SIZE)      
    c_labels = sort_c_labels(c_labels)

    c_labels_set = set(c_labels)
    if -1 in c_labels_set: c_labels_set.remove(-1) # -1 is noise
    print('Found' , len(c_labels_set), 'clusters ...')
    job.meta['NUM_CLUSTERS'] = len(c_labels_set)    
    job.save_meta()
    
    img_plt = plt_scatter(feat, c_labels, output_dir=OUTPUT_DIR, 
                plt_name='_{}_{}.png'.format('hdbscan', dim_reduce_method), pltshow=False)
    ae.tb.add_image(tag='_{}_{}'.format('hdbscan', dim_reduce_method), 
                    img_tensor=img_plt, 
                    global_step = ae.EPOCH, dataformats='HWC')
                    
    job.meta['progress_msg'] = 'Saving <b>[ {} ]</b> images ...'.format(imgs.shape[0])
    job.save_meta()
    print('Saving images to client/static/imgs...')
    if os.path.exists(app.config['IMG_DIR']): # Clearing img dir
        shutil.rmtree(app.config['IMG_DIR'])    
    os.makedirs(app.config['IMG_DIR'])
    for i, img in enumerate(imgs):
        save_image(img.view(-1, 1, 28, 28), app.config['IMG_DIR']+'/{}.png'.format(i))
 
    print('Saving processed ae feat ...')      # Saving feat to json
    np.save(os.path.join(OUTPUT_DIR, '_feat.npy'), feat)
    print('Saving c_labels ...')  
    np.save(os.path.join(OUTPUT_DIR, '_c_labels.npy'), c_labels)
    
    return feat, c_labels, imgs



def som(img_idx, c_label, dims, som_mode, num_refresh):
    img_idx = np.array(img_idx)
    job = get_current_job()
    job.meta['NUM_IMGS'] = len(img_idx)
    job.save_meta()
    
    OUTPUT_DIR = app.config['OUTPUT_DIR'] # returns the  OUTPUT DIR of the latest models by default
    feat = np.load( os.path.join(OUTPUT_DIR, '_feat.npy'))
    c_labels = np.load( os.path.join(OUTPUT_DIR, '_c_labels.npy'))
     
    img_idx = np.array(img_idx)
    lut = dict(enumerate(list(img_idx)))
    data = feat[img_idx]
    
    som_path = os.path.join(OUTPUT_DIR, '_som_{}.npy'.format(c_label))
    if som_mode == 'new':  # Declare new SOM 
        if len(img_idx) > 2000:
            iter = 3000
            lr = 0.2  
        elif len(img_idx) > 1000:
            iter = 2000
            lr = 0.2  
        elif len(img_idx) > (dims[0]*dims[1]):
            iter = 1000
            lr = 0.1
        else:
            iter = 500
            lr = 0.1
        som = SOM(data=data, dims=dims, n_iter = iter, lr_init=lr, job=job)
    elif som_mode == 'update': # Update net
        iter = 100
        lr = 0.001
        if len(img_idx) <= (dims[0]*dims[1]):   # Get whole grid when img_idx getting sparse
            iter = 500
            lr = 0.1
        som = SOM(data=data, dims=dims, n_iter = iter, lr_init=lr, net_path=som_path, job=job)
    elif som_mode == 'switch':
        iter = 1
        lr = 0.0001 
        som = SOM(data=data, dims=dims, n_iter = iter, lr_init=lr, net_path=som_path, job=job)
    
    job.meta['MAX_ITER'] = iter
    job.meta['LR'] = lr
    job.save_meta()

    print('Ordering image grid with Self Organising Map {} {}...'.format(c_label, len(img_idx)))
    print('iter: {} lr: {}\n'.format(iter, lr))
    net = som.train()
    net_w = np.array([net[x, y, :] for x in range(net.shape[0]) for y in range(net.shape[1])])
    img_grd_idx, _ = pairwise_distances_argmin_min(net_w, data)
    img_grd_idx = np.array([lut[i] for i in img_grd_idx])   # Remapping to img_idx indices

    # if not os.path.exists(som_path):  
    # Plotting HDBSCAN SOM grid     
    if som_mode != 'switch':    # don't plot if switching between clusters or filter mode
        # ae, feat_ae, labels, imgs = load_model(OUTPUT_DIR)  
        # img_plt = plt_scatter([feat, feat[img_grd_idx]], c_labels, colors=['blue'], 
        #                         output_dir=OUTPUT_DIR, 
        #                         plt_name='_{}_som_2D_{}_refresh={}_iter={}_lr={}.png'.format('hdbscan', c_label, num_refresh, iter, lr), 
        #                         pltshow=False, 
        #                         plt_grd_dims=dims)
        # ae.tb.add_image(tag='_{}_som_2D_{}_refresh={}_iter={}_lr={}.png'.format('hdbscan', c_label, num_refresh, iter, lr), 
        #                                 img_tensor=img_plt, 
        #                                 global_step = ae.EPOCH, dataformats='HWC')                        
        # # Saving image grid
        # img_grd = imgs[img_grd_idx].view(-1, 1, 28, 28)
        # save_image(img_grd, OUTPUT_DIR+'/_img_grd_som_2D_{}_refresh={}_iter={}_lr={}.png'.format(c_label, num_refresh, iter, lr), nrow=dims[1]) # nrow = num in row
        # ae.tb.add_image(tag='_img_grd_som_2D_{}_refresh={}_iter={}_lr={}.png'.format(c_label, num_refresh, iter, lr), 
        #                         img_tensor=make_grid(img_grd, nrow=dims[1]), 
        #                         global_step = ae.EPOCH)

        print('Saving SOM weights ...')
        som = {'net': net}
        np.save(som_path, som)
                            
    return img_grd_idx, c_label



                            
# Helper functions for cluster()
def load_model(output_dir):
    ae = AutoEncoder()  
    # dataset = FilteredMNIST(output_dir=output_dir)
    dataset = ImageBucket(output_dir=output_dir)
    model_path = ae.load_model(output_dir=output_dir)
    dataset.test += dataset.train   # Get all the data, eval_model loads dataset.test
    _, feat, labels, imgs = ae.eval_model(dataset=dataset, output_dir=output_dir)
    return ae, feat, labels, imgs

def umap(feat_ae, dim_reduce, min_cluster_size):
    umap = UMAP(n_components=dim_reduce, n_neighbors=min_cluster_size, min_dist=0.1,
                    random_state=SEED, transform_seed=SEED)
    feat = umap.fit_transform(feat_ae)
    feat = MinMaxScaler().fit_transform(feat) 
    return feat

def tsne(feat_ae, dim_reduce):
    tsne = TSNE(perplexity=30, n_components=dim_reduce, init='pca', n_iter=1000, random_state=SEED)
    feat = tsne.fit_transform(feat_ae)
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

def clear_output(output_dir):
    print('Clearing old output ...')
    files = glob.glob(output_dir+'/_*')
    for f in files:
        os.remove(f)

def clear_upload_folder(label):
    if os.path.exists(os.path.join(app.config['DATASET_DIR'], label)):
        shutil.rmtree(app.config['UPLOAD_DIR'])


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


# TODO: Test for allowed file ext 
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
