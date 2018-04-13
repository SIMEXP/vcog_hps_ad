__author__ = 'Christian Dansereau'

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
import numpy as np
from sklearn.preprocessing import scale
from proteus.matrix import tseries as ts

def hclustering(data, t):
    # Normalize features
    data_ = scale(data, axis=0, with_mean=True, with_std=True, copy=True)
    # Normalize observation
    #data_ = scale(data_, axis=1, with_mean=True, with_std=True)

    #row_dist = pd.DataFrame(squareform(pdist(data, metric='euclidean')))
    #row_dist = np.corrcoef(data)
    #row_dist = data
    row_clusters = linkage(data_, method='ward')
    ind = fcluster(row_clusters, t, criterion='maxclust')
    return ind

'''
def part_inter(m,ind):
    n_cluster = np.max(ind)
    new_m = np.identity(n_cluster, dtype=None)
    for i1 in range(0, n_cluster-1):
        for i2 in range(i1+1, n_cluster):
            new_m[i1, i2] = np.mean(m[ind==i1+1,:][:,ind==i2+1])

    new_m += np.triu(new_m,1).T
    return new_m
'''

def part(m,ind):
    # This function calculate the new partition with the intracluster values
    n_cluster = np.max(ind)
    new_m = np.identity(n_cluster, dtype=None)
    for i1 in range(0, n_cluster):
        for i2 in range(i1, n_cluster):
            new_m[i1, i2] = np.mean(m[ind==i1+1,:][:,ind==i2+1])

    new_m += np.triu(new_m,1).T
    return new_m

def projectmat(source,ind):
    '''
    source: is a matrix
    part  : is a partition that we whant to map the data of the source in it
    '''
    n_cluster = np.max(ind)
    n = len(ind)
    new_m = np.zeros((n,n))
    for i1 in range(0,n_cluster):
        for i2 in range(0,n_cluster):
            index_2modif = getCoordo(ind,i1+1,i2+1,n)
            new_m[index_2modif] = source[i1,i2]
    return new_m


def getCoordo(ind,i1,i2,n):
    new_m1 = np.zeros((n,n))
    new_m2 = np.zeros((n,n))
    new_m1[ind==i1,:] = 1
    new_m2[:,ind==i2] = 1
    return np.where(np.logical_and(new_m1,new_m2))

def ind2matrix(ind):
    n_cluster = np.max(ind)
    new_m = np.zeros((len(ind), len(ind)))
    for i1 in range(0, len(ind)):
        same_id = np.where(ind == ind[i1])
        new_m[i1, same_id] = ind[i1]
        new_m[same_id,i1] = ind[i1]
    return new_m

def order(ind):
    order_idx = []
    for i in range(0,max(ind)):
        l = (ind == i+1)
        order_idx = order_idx + np.where(l)[0].tolist()
    #tmp_ind = ind.copy()
    #return tmp_ind.sort()
    return np.array(order_idx)

def ordermat(m,ind):
    order_idx = order(ind)
    return m[order_idx,:][:,order_idx]

def ordermat_auto(m):
    ind = hclustering(m, m.shape[0])
    return ordermat(m,ind)

def get_ind_high2low(template_low,template_high):
    n_cls = np.max(template_high)
    ind_low_scale = []
    for i in range(n_cls):
        ind_low_scale.append(int(np.mean(template_low[template_high==i+1])))

    return np.array(ind_low_scale)

def get_mask_high2low(template_mask,template_high):
    masked_template = template_mask*template_high
    n_cls = np.max(template_high)
    ind_low_scale = []
    for i in range(n_cls):
        ind_low_scale.append(np.nan_to_num(np.mean(masked_template[masked_template==i+1])))

    return np.array(ind_low_scale)

def getWindowCluster(timeseries,nclusters=12,window_size=20):
    binary_mat = np.array([])
    for i in range(0,timeseries.shape[1]-window_size+1,1):

        clust_ind = hclustering(timeseries[:,i:i+window_size],nclusters)

        tmp_mat = np.array(ind2matrix(clust_ind)>0,dtype=int)

        if i==0:
            binary_mat = ts.mat2vec(tmp_mat)[np.newaxis,:]
        else:
            binary_mat = np.vstack((binary_mat, ts.mat2vec(tmp_mat)[np.newaxis,:]))

    print(binary_mat.shape)
    return binary_mat

def getWindows(timeseries,window_size=20,vectorize=True):
    conn_mat = []
    for i in range(0,timeseries.shape[1]-window_size+1,1):
        if vectorize:
            tmp_conn_mat = np.corrcoef(timeseries[:,i:i+window_size])
            if i==0:
                conn_mat = ts.mat2vec(tmp_conn_mat)[np.newaxis,:]
            else:
                conn_mat = np.vstack((conn_mat, ts.mat2vec(tmp_conn_mat)[np.newaxis,:]))
        else:
            tmp_conn_mat = np.corrcoef(timeseries[:,i:i+window_size])
            conn_mat.append(tmp_conn_mat)
    conn_mat = np.array(conn_mat)
    #print conn_mat.shape
    return conn_mat

# Test functions
def test_ind2matrix():
    ind = np.array([1,2,3,1])
    assert np.all(ind2matrix(ind) == np.array([[ 1.,  0.,  0.,  1.],
       [ 0.,  2.,  0.,  0.],
       [ 0.,  0.,  3.,  0.],
       [ 1.,  0.,  0.,  1.]]))
