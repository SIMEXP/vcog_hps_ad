__author__ = 'Christian Dansereau'

import numpy as np
from mvpa2.datasets import *
from sklearn.model_selection import StratifiedKFold
from mvpa2.measures import irelief as irelief_mvpa

def modelpred(x,y):
    # Preporcessing 
    #scaler = preprocessing.StandardScaler().fit(x)
    #x_scaled = scaler.transform(x)
    x_scaled = x
    #encoder = preprocessing.LabelEncoder()
    #encoder.fit(y)

    # Feature selection
    w = fselect.irelief_cross(x_scaled,y,10)
    #candidat_f = fselect.nBest(w,108)
    fselect.threhold_std()

    # grid search and SVM
    clf = get_opt_model(x_scaled[:,candidat_f],y)
    #clf = svm.SVC(kernel='rbf', class_weight='balanced')
    #clf = plib.grid_search(clf, x_scaled[:,candidat_f], y, n_folds=10, verbose=True)
    #clf.fit(x_scaled[:,candidat_f],y)
    #clf = plib.classif(x_scaled[:,candidat_f],y)
    return clf, candidat_f, w

def get_opt_model_features_nbest(x,y,w):
    n = len(fselect.threhold_std(w,1))
    best_score = 0
    best_clf = []
    best_i = 0
    all_nfeatures = []
    all_scores = []
    for i in range(5,int(n),20):

        # grid search and SVM
        candidat_f = fselect.nBest(w,i)
        clf = svm.SVC(kernel='linear', class_weight='balanced')
        clf.probability = True
        clf, score = plib.grid_search(clf, x[:,candidat_f], y, n_folds=10, verbose=False)
        #print score
        all_nfeatures.append(i)
        all_scores.append(score)
        if score > best_score:
            print('New best: ', best_score)
            best_score = score
            best_clf = clf
            best_i = i

    return best_clf,fselect.nBest(w,i),best_i,all_nfeatures,all_scores

def get_opt_model_features_std(x,y,w,std_step=0.25):
    best_score = 0
    best_clf = []
    best_i = 0
    all_nfeatures = []
    all_scores = []
    for i in np.arange(0,2.75,std_step):

        # grid search and SVM
        candidat_f = fselect.threhold_std(w,i)
        print(candidat_f.shape)
        if len(candidat_f)>0:
            clf = svm.SVC(kernel='linear', class_weight='balanced',C=0.01)
            clf.probability = True
            clf, score = plib.grid_search(clf, x[:,candidat_f], y, n_folds=10, verbose=False)
            #print score
            all_nfeatures.append(i)
            all_scores.append(score)
            if score > best_score:
                best_score = score
                best_clf = clf
                best_i = i
                print('New best: ', best_score)

    return best_clf,fselect.threhold_std(w,best_i),best_i,all_nfeatures,all_scores

def sv_metric(n,nsv):
    return nsv/float(n)
    #return (n-nsv)/float(n) #lower the n sv is greater the score

def get_opt_model_features_std_sv(x,y,w,alpha=0.8):
    best_score = 0
    best_clf = []
    best_i = 0
    best_nsv = 0
    all_nfeatures = []
    all_scores = []
    for i in np.arange(1,2.75,0.25):

        # grid search and SVM
        candidat_f = fselect.threhold_std(w,i)
        if len(candidat_f)>0:
            clf = svm.SVC(kernel='linear', class_weight='balanced')
            clf.probability = True
            clf, score = plib.grid_search(clf, x[:,candidat_f], y, n_folds=10, verbose=False)
            #print score
            all_nfeatures.append(i)
            all_scores.append(score)
            if best_score < (score - alpha*sv_metric(x.shape[0],np.sum(clf.n_support_))):
                best_score = score-alpha*sv_metric(x.shape[0],np.sum(clf.n_support_))
                best_clf = clf
                best_i = i
                best_nsv = clf.n_support_
                print('New best: ', best_score)

    return best_clf,fselect.threhold_std(w,best_i),best_i,all_nfeatures,all_scores,best_nsv



def nBest(w,n,verbose=False):
    sorted_avg = []
    if w.ndim > 1:
        sorted_avg = np.argsort(w.mean(axis=0))
    else:
        sorted_avg = np.argsort(w)
    if verbose:
        print(sorted_avg)
    
    return sorted_avg[-n:]

def threhold_std(w,nstd=2, verbose=False):
    candidates_idx = []
    if w.ndim > 1:
        w_scores = (w.mean(axis=0).std()*nstd)+w.mean()
        candidates_idx = np.argwhere(w.mean(axis=0) >= w_scores)
    else:
        w_scores = (w.std()*nstd)+w.mean()
        candidates_idx = np.argwhere(w >= w_scores)
    
    if verbose:
        print(candidates_idx)
    
    return candidates_idx[:,0]


def irelief(x,y):
    #format the dataset
    ds = dataset_wizard(x, targets=y)
    # irelief 
    fs = irelief_mvpa.IterativeReliefOnline()
    ds2 = fs._call(ds)
    return ds2.samples.flatten().flatten()

def irelief_cross(x,y,folds=10,verbose=True):

    print("iRelief ...")
    #format the dataset
    w=[]
    if folds == 1:
        return irelief(x,y)
    else:
        skf = StratifiedKFold(y, folds)
        for train, test in skf:
            ds = dataset_wizard(x[train,:], targets=y[train])
            samps = ds.samples
            if verbose:
                print("iRelief, sample size: ", samps.shape[:2])
            w_tmp = irelief(x[train,:], y[train])
            #print(samps.shape[:2])
            if len(w)>0:
                w = np.vstack((w,w_tmp))
            else:
                w = w_tmp
    return w

def near(X,xi):
    # nearest point to x in P with the same label
    print(X)
    print(xi)
    idx = np.abs(X - xi).argmin()
    return idx

def norm(x,order=2):
    return np.linalg.norm(x, ord=order)   

def excludeIdx(idx,X,Y):
    return  X[idx,:], np.delete(X,idx,axis=0), np.delete(Y,idx,axis=0)

def normw(W,z):
    return norm(W*z,2)
   
def margin(idx,X,Y,W,label_hit = 1,label_miss = -1):
    xi,Xp,Yp = excludeIdx(idx,X,Y)
    nearhit = near(Xp[Yp == label_hit],xi)
    nearmiss = near(Xp[Yp == label_miss],xi)
    return 1/2*( normw(W,xi-Xp[nearmiss,:]) - normw(W,xi-Xp[nearhit,:]) )

def gflip(x,y):
    """
    Greedy Feature Flip (G-flip)
    Gilad-Bachrach (2004)
    """
    F = []; #Initialize the set of chosen features to the empty
    # for t = 1,2,...
    #for t in range(5):
        # pick a random permutation s of {1 . . . N} 
        #for i in s:
            # for each value i in s 

if __name__ == "__main__": # test the implementation
    X = np.array([[1.1, 2.1, 3.1, -1.0],  # first sample
        [1.2, 2.2, 3.2, 1.1],   # second sample
        [1.3, 2.3, 3.3, -1.1], # third sample
        [5.3, 5.3, 3.5, 1.0]]) # third sample  
    Y = np.array([1, -1, 1, -1])     # classes
    #nearhit(X[0,:],np.delete(X,0,axis=0),np.delete(Y,0,axis=0))
    W = np.ones(X.shape[1])
    print(margin(0,X,Y,W,1,-1))
