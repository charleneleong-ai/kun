#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, August 28th 2019, 11:17:31 am
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Wed Aug 28 2019
# -----
# Copyright (c) 2019 Victoria University of Wellington ECS
###
# Author: Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import sys
sys.path.append('..')
import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

from ae.conv_ae import ConvAutoEncoder
from utils.datasets import FilteredMNIST
import seaborn as sns
sns.set()
# CURRENT_FNAME = os.path.basename(__file__).split('.')[0]
# timestamp = datetime.now().strftime('%Y.%d.%m-%H:%M:%S')
# OUTPUT_DIR = './{}_{}_output'.format(CURRENT_FNAME, timestamp)

SEED = 489
np.random.seed(SEED)

def plot_ellipses(ax, weights, means, covars):
    for n in range(means.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(means[n], eig_vals[0], eig_vals[1],
                                  180 + angle, edgecolor='black')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(weights[n])
        ell.set_facecolor('#56B4E9')
        ax.add_artist(ell)


def plot_results(ax1, ax2, estimator, X, y, title, plot_title=False):
    ax1.set_title(title)
    ax1.scatter(X[:, 0], X[:, 1], s=5, marker='o', c=y, alpha=0.8)
    ax1.set_xlim(-1.3, 2.8)
    ax1.set_ylim(-3., 3.)
    ax1.set_xticks(())
    ax1.set_yticks(())
    plot_ellipses(ax1, estimator.weights_, estimator.means_,
                  estimator.covariances_)

    ax2.get_xaxis().set_tick_params(direction='out')
    ax2.yaxis.grid(True, alpha=0.7)
    for k, w in enumerate(estimator.weights_):
        ax2.bar(k, w, width=0.9, color='#56B4E9', zorder=3,
                align='center', edgecolor='black')
        ax2.text(k, w + 0.007, "%.1f%%" % (w * 100.),
                 horizontalalignment='center')
    ax2.set_xlim(-.6, n_components - .4)
    ax2.set_ylim(0., 1.1)
    ax2.tick_params(axis='y', which='both', left=False,
                    right=False, labelleft=False)
    ax2.tick_params(axis='x', which='both', top=False)

    if plot_title:
        ax1.set_ylabel('Estimated Mixtures')
        ax2.set_ylabel('Weight of each component')

# Parameters of the dataset
random_state, n_components, n_features = 2, 6, 2
# colors = np.array(['#0072B2', '#F0E442', '#D55E00'])

# covars = np.array([[[.7, .0], [.0, .1]],
#                    [[.5, .0], [.0, .1]],
#                    [[.5, .0], [.0, .1]]])
# samples = np.array([200, 500, 200])
# means = np.array([[.0, -.70],
#                   [.0, .0],
#                   [.0, .70]])

# mean_precision_prior= 0.8 to minimize the influence of the prior
estimators = [
    ("Finite mixture with a Dirichlet distribution\nprior and "
     r"$\gamma_0=$", BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_distribution",
        n_components=n_components, reg_covar=0, init_params='kmeans',
        max_iter=1500, mean_precision_prior=.8,
        random_state=random_state), [0.001, 1, 1000]),
    ("Infinite mixture with a Dirichlet process\n prior and" r"$\gamma_0=$",
     BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_process",
        n_components=n_components, reg_covar=0, init_params='random',
        max_iter=1500, mean_precision_prior=.8,
        random_state=random_state), [1, 1000, 100000])]

# Generate data
# rng = np.random.RandomState(random_state)
# X = np.vstack([
#     rng.multivariate_normal(means[j], covars[j], samples[j])
#     for j in range(n_components)])
# y = np.concatenate([np.full(samples[j], j, dtype=int)
#                     for j in range(n_components)])

ae = ConvAutoEncoder()  

OUTPUT_DIR = max(glob.iglob('./*/'), key=os.path.getctime)

dataset = FilteredMNIST(output_dir=OUTPUT_DIR)
model_path = ae.load_model(output_dir=OUTPUT_DIR)


# # =================== CLUSTER ASSIGNMENT ===================== #
_, feat, labels = ae.eval_model(dataset=dataset, 
                                batch_size=ae.BATCH_SIZE, 
                                epoch=ae.EPOCHS, 
                                plt_imgs=None, 
                                # scatter_plt=('tsne', ae.EPOCHS),    
                                output_dir=OUTPUT_DIR)

pca = PCA(n_components=2, random_state=SEED)
feat = pca.fit_transform(feat)
feat = StandardScaler().fit_transform(feat)    # Normalise the data
X = feat
y = labels
print(X.shape)
print(len(y))

# sns.reset_orig()

# Plot results in two different figures
for (title, estimator, concentrations_prior) in estimators:
    plt.figure(figsize=(4.7 * 3, 8))
    plt.subplots_adjust(bottom=.04, top=0.90, hspace=.05, wspace=.05,
                        left=.03, right=.99)

    gs = gridspec.GridSpec(3, len(concentrations_prior))
    for k, concentration in enumerate(concentrations_prior):
        estimator.weight_concentration_prior = concentration
        estimator.fit(X)
        plot_results(plt.subplot(gs[0:2, k]), plt.subplot(gs[2, k]), estimator,
                     X, y, r"%s$%.1e$" % (title, concentration),
                     plot_title=k == 0)


    plt.savefig('./{}_{}_{}.png'.format((title).split(' with')[0], 'encoded', 'pca'), bbox_inches='tight')
