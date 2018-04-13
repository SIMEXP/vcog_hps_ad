__author__ = 'Christian Dansereau'

import numpy as np
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing

import multiprocessing as mp
from multiprocessing import Pool
from scipy import stats
from proteus.matrix import tseries as ts
from proteus.predic import clustering as cls

def getkBest(x,y,k=0):
    y_val = np.unique(y)
    # t-test
    ttest, pval = stats.ttest_ind(x[y==y_val[0],:], x[y==y_val[1],:])
    
    # order absolute ttest scores
    sort_idx = np.argsort(np.abs(ttest)) # (small to large t-values)
    sort_idx = sort_idx[::-1] #flip the array to have a decresing list of values
    if k==0:
        # return all the idx
        return sort_idx
    else:
        #return the k larger idx
        return sort_idx[:k]

def itStability(x,y,ind,k=1,samp_ratio=0.5,nsample=100):
    '''
    A random iterative resampling of the subject to compute the stability of the selected features
    '''
    subj_idx = range(0,x.shape[0])
    stability_hr_mat = np.zeros((len(ind),len(ind)))
    for i in range(0,nsample):
        sample_idx = np.random.permutation(subj_idx)[:np.int(len(subj_idx)*samp_ratio)]
        bestidx = getkBest(x[sample_idx,:],y[sample_idx],k)
        it_vote = np.zeros(x.shape[1])
        it_vote[bestidx] = 1 # create the vector of selected features
        lr_mat = ts.vec2mat(it_vote,include_diag=True) # convert to the low resolution matrix
        hr_mat = cls.projectmat(lr_mat,ind) # remap in HR
        # Add the matrix to the main HR stability matrix
        stability_hr_mat += hr_mat

    stability_hr_mat /= nsample
    return stability_hr_mat

'''
class stability:

    def __init__(self, confounds, data):
        self.fit(data, confounds)

    def fit(self, confounds, data):
        self.reg = linear_model.LinearRegression(fit_intercept=True)
        self.reg.fit(data, confounds)

    def transform(self, confounds, data):
        # compute the residual error

        return data - self.reg.predict(confounds)

'''


