#!/usr/bin/env python
# coding: utf-8

# # Generate some fake data from a mixture of randomly placed Gaussians.

# In[8]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import sys, math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng


# In[9]:


def random_rotation(C, angle):
    invC = np.linalg.inv(C)
    R = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]]) # rotation matrix
    invC = np.dot(np.linalg.inv(R), np.dot(invC, R))
    C = np.linalg.inv(invC)
    return invC, C


# In[10]:

if __name__ == '__main__':
    
    D = 2    # number of dimensions
    if len(sys.argv) == 4:
        K = int(sys.argv[1]) # number of components
        N = int(sys.argv[2]) # number of data points
        outfile = str(sys.argv[3])
        print ('We are assuming {0} classes and making {1} data points, each {2}-dimensional'.format(K,N,D))
    else:
        sys.exit('usage: python {0} numClasses  numDatapoints  filename.csv'.format(sys.argv[0]))

    out_stem = outfile.replace('.csv','')
    out_file = out_stem + '.csv'

    prior = 0.1 + 0.1*rng.random((1,K))    # mixing coefficients
    prior[0,0] = 1.0 # ie. make one "popular" one
    prior = prior / np.sum(prior)             # normalisation
    print("prior mixture coefficients : ",prior)
    
    covariances = []
    centers = []
    for k in range(K):
        centers.append(10 * rng.normal(0.0,1.0,(D)))
        C = np.array([[0.01+0.99*rng.random(), 0.0],[0.0, 0.01+0.1*rng.random()]])
        angle = rng.random() * math.pi
        C, invC = random_rotation(C, angle)

        covariances.append(0.5*C)

    # generate samples from this mixture of Gaussians
    data = np.zeros((N,D))
    for i in range(N):
        # choose a component
        j = np.sum(rng.random() > np.cumsum(prior))
        # Now choose a data point using that component of the mixture
        x,y = rng.multivariate_normal(centers[j],covariances[j],1).T
        data[i,0] = x
        data[i,1] = y

    # perhaps we could normalise the data, so it's zero mean, and unit variance, overall.
    data = (data - data.mean(0)) / data.std(0)



    # show the samples as a scatter plot
    plt.scatter(data[:,0], data[:,1], marker='o',s=.5,linewidths=None,alpha=1.0)
    plt.axis('equal')    
    plt.draw()


    # If it's 2-d, save it as a PNG file, just for kicks.
    if D==2:
        out_imagename = out_stem + '.png'
        plt.savefig(out_imagename)
        print ('saved image %s' % out_imagename)

    # Now write a datafile consisting of a matrix, in which each row is a
    # training item. Ground truth is lost in this file: the true class is not
    # written, just the vector...

    # write the samples to a file
    np.savetxt(out_file, data, fmt="%12.6G",)
    print ('wrote data file %s' %out_file)

