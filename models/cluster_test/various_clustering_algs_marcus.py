
#!/usr/bin/env python
# coding: utf-8

# Cluster data using various methods.
#
# Heavily based on https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
# See that page for nice explanations too.
#
# Expects to read a datafile consisting of a matrix in which each row is a training item.

import sys, math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import sklearn.cluster as cluster
import time
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 50, 'linewidths':1}


def plot_clusters(outfile, data, algorithm, args, kwds):
    fig = plt.figure()
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=14)
    #plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    plt.savefig(outfile)
    print ( '\n  saved image ',outfile )
    plt.close(fig)
    return    


if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Three groups of system parameters.
    parser.add_argument('-f','--infile', action="store", default='data.csv', help='input data as a .csv file')
    args = parser.parse_args()
    out_stem = args.infile.split('.')[0]

    # Read in some data from a csv file
    data = np.genfromtxt(args.infile, float) #, unpack=True)
    NUM_DATA_ITEMS,NUM_DATA_DIM = data.shape
    
    plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.savefig(out_stem+'_RAW.png')
    
    
    ### See this with great pithy explanations at 
    ### https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    import hdbscan
    plot_clusters(out_stem+'_HDBSCAN.png',data, hdbscan.HDBSCAN, (), {'min_cluster_size':15}) 
    
    plot_clusters(out_stem+'_Kmeans.png',data, cluster.KMeans, (), {'n_clusters':6})

    plot_clusters(out_stem+'_MeanShift.png', data, cluster.MeanShift, (0.175,), {'cluster_all':False})
    
    plot_clusters(out_stem+'_SpectralClustering.png',data, cluster.SpectralClustering, (), {'n_clusters':6})
    
    plot_clusters(out_stem+'_AffinityProp.png',data, cluster.AffinityPropagation, (), {'preference':-5.0, 'damping':0.95})
    
    plot_clusters(out_stem+'_AgglomerativeClustering.png',
                  data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'ward'})

    
