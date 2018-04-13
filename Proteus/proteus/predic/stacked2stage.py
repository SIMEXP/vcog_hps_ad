__author__ = 'Christian Dansereau'

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

import multiprocessing
import time

class StackedPrediction:
    '''
    2 Level prediction
    '''

    def makeNewModel(self,x,y):
        clf = LogisticRegression(C=1.,class_weight='balanced',penalty='l1')
        #clf = LogisticRegressionCV(Cs=10,class_weight='balanced',penalty='l1',solver='liblinear')
        #clf = LinearSVC(C=1.,class_weight='balanced')
        param_grid = dict(C=(10**np.arange(0.,2.,.25)))
        gridclf = GridSearchCV(clf, param_grid=param_grid, cv=StratifiedKFold(y,n_folds=4), n_jobs=-1,scoring='precision_weighted')
        #gridclf = GridSearchCV(clf, param_grid=param_grid, cv=StratifiedKFold(y,n_folds=4), n_jobs=-1,scoring='precision')
        #gridclf = GridSearchCV(clf, param_grid=param_grid, cv=StratifiedKFold(y,n_folds=4), n_jobs=-1,scoring='accuracy')
        gridclf.fit(x,y)
        #clf.fit(x,y)
        return gridclf.best_estimator_, gridclf.best_score_
        #return clf,0.0

    def fitMulti(self,x,y,skip_init=False):
        if not skip_init:
            self.clf_list = []
            self.scores_stage1 = []

        for ii in range(x.shape[1]):
            if not skip_init:
                model,score = self.makeNewModel(x[:,ii,:],y)
                self.scores_stage1.append(score)
                self.clf_list.append(model)
            else:
                self.clf_list[ii].fit(x[:,ii,:],y)

        if not skip_init:
            print self.scores_stage1


    def predictMulti(self,x):
        pred = []
        for ii in range(len(self.clf_list)):
            if len(x.shape) == 2:
                #pred.append(self.clf_list[ii].predict(x[ii,:].reshape(1,-1)))
		pred.append(self.clf_list[ii].decision_function(x[ii,:].reshape(1,-1)))
            else:
                #pred.append(self.clf_list[ii].predict(x[:,ii,:]))
		pred.append(self.clf_list[ii].decision_function(x[:,ii,:]))
       #     print self.clf_list[ii]
       #     print (self.clf_list[ii].coef_**2.).sum()
        pred = np.array(pred)
        if len(pred.shape) == 2:
            pred = np.swapaxes(pred,0,1)
        else:
            pred = np.swapaxes(pred,1,2)[:,:,0]
        return pred

    def fit(self,x,y):
        # Stage 1
        #hm = self.estimate_hitmiss(x, y)
        self.fitMulti(x,y)
        hm,ref_labels = self.predict_estimate(x,y)
        print 'hm shape: ',hm.shape
        #hm = self.predictMulti(x)

        print (hm == np.tile(ref_labels,(hm.shape[1],1)).T).mean(axis=0)
        # Stage 2
        clf2 = LogisticRegression(C=1.,class_weight='balanced',penalty='l2')
        param_grid = dict(C=(10**np.arange(0.,2.,0.25)))

        gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedKFold(ref_labels,n_folds=4), n_jobs=-1,scoring='accuracy')
        gridclf.fit(hm,ref_labels)
        gridclf.grid_scores_
        self.clf_stack = gridclf.best_estimator_
        print 'stack ',self.clf_stack
        print 'stack SOS  ',(self.clf_stack.coef_**2.).sum()

    def cv(self,X,y,gs=4):
        k=1
        skf = StratifiedKFold(y, n_folds=gs)
        scores = []
        for train_index, test_index in skf:
            print('Fold: '+str(k)+'/'+str(gs))
            k+=1
            # train
            self.fit(X[train_index], y[train_index])
            # test
            y_pred = self.predict(X[test_index])
            scores.append((y_pred==y[test_index]).sum()/(1.*y[test_index].shape[0]))
            print classification_report(y[test_index], y_pred)

        scores = np.array(scores)
        return [scores,scores.mean(),scores.std()]

    def predict(self,x):
        l1_results = self.predictMulti(x)
        return self.clf_stack.predict(l1_results)


        #y_pred1 = self.clf1.predict(xw)
        #y_pred2 = self.clf2.decision_function(xw)
        #return np.array([y_pred1,y_pred2]).T
    def score(self,x,y):
        l1_results = self.predictMulti(x)
        l2_results = self.clf_stack.predict(l1_results)
        return (l2_results==y).mean()

    def predict_estimate(self,x,y,kf=4):
        # Perform a kfold CV to estimate the actual prediction error
        skf = StratifiedKFold(y, n_folds=kf)
        results = []
        ref_labels = []
        for train_index, test_index in skf:
            self.fitMulti(x[train_index,:,:],y[train_index],skip_init=True)
            results.append(self.predictMulti(x[test_index,:,:]))
            ref_labels.extend(y[test_index])
        ref_labels = np.array(ref_labels)
        results = np.vstack(results)

        self.fitMulti(x,y,skip_init=True)
        return results, ref_labels


    def estimate_hitmiss(self,x,y):
        # Perform a LOO to estimate the actual HM
        label=1
        hm_results = []
        predictions =[]
        for i in range(len(y)):
            train_idx = np.array(np.hstack((np.arange(0,i),np.arange(i+1,len(y)))),dtype=int)
            self.fitMulti(x[train_idx,:,:],y[train_idx])
            hm_results.append((self.predictMulti(x[i,:,:]) == y[i]).astype(int))
            #predictions.append(clf.predict(x[i,:,:].reshape(1,-1)))

        #predictions = np.array(predictions)
        hm_results = np.array(hm_results)
        hm_results = np.swapaxes(hm_results,1,2)[:,:,0]

        #print hm_results.shape
        self.fitMulti(x,y)
        return hm_results#, predictions[:,0]

