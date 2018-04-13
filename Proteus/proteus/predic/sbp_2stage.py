__author__ = 'Christian Dansereau'

import numpy as np
from proteus.predic import prediction
from proteus.predic import subtypes
from sklearn.svm import SVC,l1_min_c
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut, StratifiedKFold,StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from nistats import glm as nsglm
import statsmodels.stats.multitest as smm
import multiprocessing
import time

def compute_loo_parall((net_data_low_main,y,confounds,n_subtypes,train_index,test_index)):
    my_sbp = sbp()
    my_sbp.fit(net_data_low_main[train_index,...],y[train_index],confounds[train_index,...],n_subtypes,verbose=False)
    tmp_scores = my_sbp.predict(net_data_low_main[test_index,...],confounds[test_index,...])
    return np.hstack((y[test_index],tmp_scores[0][0],tmp_scores[1][0]))

class SBP:
    '''
    Pipeline for subtype base prediction
    '''
    def fit(self,net_data_low_main,y,confounds,n_subtypes,n_subtypes_l1=3,flag_feature_select=True,extra_var=[],verbose=True):
        self.verbose = verbose
        ### regress confounds from the connectomes
        #net_data_low = net_data_low_main.copy()
        #cf_rm = prediction.ConfoundsRm(confounds,net_data_low.reshape((net_data_low.shape[0],net_data_low.shape[1]*net_data_low.shape[2])))
        #net_data_low_tmp = cf_rm.transform(confounds,net_data_low.reshape((net_data_low.shape[0],net_data_low.shape[1]*net_data_low.shape[2])))
        #net_data_low = net_data_low_tmp.reshape((net_data_low_tmp.shape[0],net_data_low.shape[1],net_data_low.shape[2]))
        self.scale_ref = net_data_low_main.mean(0).mean(1)
        #net_data_low = self.norm_subjects(net_data_low_main)
        self.cf_rm = prediction.ConfoundsRm(confounds,net_data_low_main)
        net_data_low = self.cf_rm.transform(confounds,net_data_low_main)
        #net_data_low += self.cf_rm.intercept()


        ### compute the subtypes
        if self.verbose: start = time.time()
        st_ = subtypes.clusteringST()
        st_.fit(net_data_low,n_subtypes_l1)
        xw = st_.transform(net_data_low)
        xw = np.nan_to_num(xw)

        print 'xw sub data',xw[0,:] 
        self.st_l2 = subtypes.clusteringST()
        self.st_l2.fit(net_data_low,n_subtypes)
        xwl2 = self.st_l2.transform(net_data_low)
        xwl2 = np.nan_to_num(xwl2)
        #xwl2 = np.hstack((xwl2,confounds))
        #xw = np.hstack((age_var,xw))
        if self.verbose: print("Compute subtypes, Time elapsed: {}s)".format(int(time.time() - start)))

        ### feature selection
        if flag_feature_select:
            if verbose: start = time.time()
            contrast = np.hstack(([0,1],np.repeat(0,confounds.shape[1])))#[0,1,0,0,0]
            x_ = np.vstack((np.ones_like(y),y,confounds.T)).T

            labels, regression_result  = nsglm.session_glm(np.array(xw),x_)
            cont_results = nsglm.compute_contrast(labels,regression_result, contrast,contrast_type='t')
            pval = cont_results.p_value()
            results = smm.multipletests(pval, alpha=0.01, method='fdr_bh')
            w_select = np.where(results[0])[0]
            #w_select = w_select[np.argsort(pval[np.where(results[0])])]
            if len(w_select)<10:
                w_select = np.argsort(pval)[:10]
            else:
                w_select = w_select[np.argsort(pval[np.where(results[0])])]
        else:
            # Cancel the selection
            w_select = np.where(xw[0,:]!=2)[0]

        #w_select = get_stable_w(xw[train_index,:],y_tmp[train_index],confounds[train_index,:],6)
        # Cancel the selection
        #w_select = np.where(results[0]!=-1)[0]
        #print("Feature selected: {})".format(w_select))

        ### Include extra covariates
        if len(extra_var)!=0:
            all_var = np.hstack((xw[:,w_select],extra_var))
        else:
            all_var = xw[:,w_select]
        if self.verbose: print("Feature selection, Time elapsed: {}s)".format(int(time.time() - start)))

        ### prediction model
        if self.verbose: start = time.time()
        tlp = TwoLevelsPrediction()
        tlp.fit(all_var,xwl2,y,model_type='svm',verbose=self.verbose)
        if self.verbose: print("Two Levels prediction, Time elapsed: {}s)".format(int(time.time() - start)))

        ### save parameters
        self.median_template = np.median(net_data_low,axis=0)
        self.st = st_
        self.w_select = w_select
        self.tlp = tlp

    def predict(self,net_data_low_main,confounds,extra_var=[]):
        ### regress confounds from the connectomes
        #net_data_low = net_data_low_main.copy()

        #net_data_low = net_data_low_tmp.reshape((net_data_low_tmp.shape[0],net_data_low.shape[1],net_data_low.shape[2]))
        #scale_new = net_data_low_main.mean(0).mean(1)
        #scaling_factor = self.scale_ref / scale_new
        #tmp_data = np.swapaxes(np.swapaxes(net_data_low_main,1,2)*scaling_factor,1,2)
        #net_data_low = self.norm_subjects(net_data_low_main)
        net_data_low = self.cf_rm.transform(confounds,net_data_low_main)
        #net_data_low += self.cf_rm.intercept()
        ### subtypes w estimation
        self.xw = self.st.transform(net_data_low)
        self.xw = np.nan_to_num(self.xw)
        self.xwl2 = self.st_l2.transform(net_data_low)
        self.xwl2 = np.nan_to_num(self.xwl2)
        #self.xwl2 = np.hstack((self.xwl2,confounds))
        ### Include extra covariates
        if len(extra_var)!=0:
            all_var = np.hstack((self.xw[:,self.w_select],extra_var))
        else:
            all_var = self.xw[:,self.w_select]

        ### prediction model
        return self.tlp.predict(all_var,self.xwl2)

    def score(self,net_data_low_main,y,confounds,extra_var=[]):
        res = self.predict(net_data_low_main,confounds,extra_var)
        l1_y_pred = res[:,0]
        risk_mask = res[:,1]>0
        right_cases = accuracy_score(y[risk_mask],res[risk_mask,0])
        left_cases = accuracy_score(y[~risk_mask],res[~risk_mask,0])
        return accuracy_score(y,l1_y_pred),left_cases,right_cases

    def norm_subjects(self,data,ref=[]):
        if len(data.shape) == 2:

            ref_avg_rmaps = ref.mean()
            avrg_rmaps = data.mean(1)
            scaling_factor = ref_avg_rmaps / avrg_rmaps
            return data*scaling_factor.reshape(-1,1)
        else:
            ref_avg_rmaps = self.scale_ref
            avrg_rmaps = data.mean(2)
            scaling_factor = ref_avg_rmaps / avrg_rmaps
            return np.swapaxes(np.swapaxes(data,0,2)*np.swapaxes(scaling_factor,0,1),0,2)

    def estimate_acc(self,net_data_low_main,y,confounds,n_subtypes,verbose=False):

        sss = LeaveOneOut(len(y))
        # scores: y, y_pred, decision_function
        self.scores = []
        k=0
        for train_index, test_index in sss:
            k+=1
            print('Fold: '+str(k)+'/'+str(len(y)))
            self.fit(net_data_low_main[train_index,...],y[train_index],confounds[train_index,...],n_subtypes=n_subtypes,verbose=False,flag_feature_select=False)
            tmp_scores = self.predict(net_data_low_main[test_index,...],confounds[test_index,...])
            self.scores.append(np.hstack((y[test_index],tmp_scores[0][0],tmp_scores[0][1])))
        self.scores = np.array(self.scores)
        return self.scores

    def estimate_acc_multicore(self,net_data_low_main,y,confounds,n_subtypes,verbose=False):
        taskList_loo = []
        sss = LeaveOneOut(len(y))
        # scores: y, y_pred, decision_function
        self.scores = []
        k=0
        for train_index, test_index in sss:

            taskList_loo.append((net_data_low_main,y,confounds,n_subtypes,train_index,test_index))

        pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 2)) #Don't use all my processing power.
        r2 = pool.map_async(compute_loo_parall, taskList_loo, callback=self.scores.append)  #Using fxn "calculate", feed taskList, and values stored in "results" list
        r2.wait()
        pool.terminate()
        pool.join()
        self.scores = np.array(self.scores)
        return self.scores

class TwoLevelsPrediction:
    '''
    2 Level prediction
    '''

    def fit(self,xw,xwl2,y,gs=4,model_type='logit',verbose=True):
        self.verbose = verbose
        if model_type=='logit':
            clf = LogisticRegression(C=1,class_weight='balanced',penalty='l2',max_iter=300)
        else:
            #clf = SVC(kernel='linear', class_weight='balanced', C=.1,probability=False)
            clf = SVC(C=1.,cache_size=500,kernel='linear',class_weight='balanced',probability=False)
            #clf = RandomForestClassifier(n_estimators=500,class_weight='balanced')
        '''
        # wrapper feature selection
        rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(y, 3), scoring='f1')#accuracy
        rfecv.fit(xw, y)
        print("Optimal number of features : %d" % rfecv.n_features_)
        print("ids: {}".format((rfecv.ranking_<=5).sum()))
        print rfecv.grid_scores_
        self.rfecv = rfecv
        if rfecv.support_.sum()>10:
            self.w_select = rfecv.support_
        else:
            self.w_select = rfecv.ranking_<=10
        '''
        #xw = [:,self.w_select]

        #self.mask_selection = (np.ones((1,xw.shape[1]))==1)[0,:]
        ## Optimize the hyper parameters
        # Stage 1
        #param_grid = dict(C=(np.array([5,3,1])))
        '''
        if model_type=='logit':
            param_grid = dict(C=(10**np.arange(1.,-2.,-0.5)))
            #param_grid = dict(C=(np.logspace(-.2, 1., 15)))
            #param_grid = dict(C=(np.arange(3,1,-0.5)))
        else:
            param_grid = dict(C=(np.arange(3.5,0.,-0.5)))
            param_grid = dict(C=(1.,1.00001))
            #param_grid = dict(C=(np.logspace(-1.5, 0, 10)))
            #param_grid = dict(C=(np.arange(2.,0.5,-0.05)))
            #param_grid = dict(C=(np.array([0.01, 0.1, 1, 10, 100, 1000])))

        gridclf = GridSearchCV(clf, param_grid=param_grid, cv=StratifiedKFold(y,n_folds=gs), n_jobs=-1,scoring='accuracy')
        gridclf.fit(xw,y)
        self.clf1 = gridclf.best_estimator_
        '''
        self.clf1 = clf
        self.clf1.fit(xw,y)
        if self.verbose:
            print self.clf1
            print self.clf1.coef_
            #print self.clf1.feature_importances_
        #hm_y,y_pred_train = self.estimate_hitmiss(xw,y)
        hm_y,proba = self.suffle_hm(xw,y,gamma=.8,n_iter=100)
        #hm_y = self.clf1.predict(xw)
        print 'Stage 2'
        #Stage 2
        min_c = l1_min_c(xwl2,hm_y,loss='log')
        print 'minimum c: ',min_c
        #clf2 = LogisticRegression(C=10**0.1,class_weight=None,penalty='l2',solver='sag')
        #clf2 = LogisticRegression(C=1,class_weight=None,penalty='l2',solver='sag',max_iter=300)
        #clf2 = LinearSVC(class_weight='balanced',penalty='l1',dual=False)
        clf2 = LogisticRegression(C=1.,class_weight='balanced',penalty='l2',solver='liblinear',max_iter=300)
        #clf2 = LogisticRegression(C=1.,class_weight='balanced',penalty='l2',solver='sag',max_iter=300)
        #clf2 = SVC(C=1.,cache_size=500,kernel='linear',class_weight='balanced')
        #clf2 = RandomForestClassifier(n_estimators=500,class_weight='balanced',oob_score=True)
        #param_grid = dict(C=(10**np.arange(1.,-2.,-0.5)))
        #param_grid = dict(C=(np.arange(3,1,-0.5)))
        #param_grid = dict(C=(np.logspace(-0.5, 2., 30)))
        #param_grid = dict(C=(np.logspace(1., 1.6, 30)))
        param_grid = dict(C=(np.logspace(-.2, 1., 15)))
        #param_grid = dict(C=(np.logspace(-.15, 1., 15)))
        #param_grid = dict(C=(np.logspace(np.log10(min_c), 0., 15)))
        #param_grid = dict(C=(1,1.0001)) 
        # 2 levels balancing
        '''
        new_classes = np.zeros_like(y)
        new_classes[(y==0) & (hm_y==0)]=0
        new_classes[(y==1) & (hm_y==0)]=1
        new_classes[(y==0) & (hm_y==1)]=2
        new_classes[(y==1) & (hm_y==1)]=3

        tmp_samp_w = len(new_classes) / (len(np.unique(new_classes))*1. * np.bincount(new_classes))
        tmp_samp_w = (1.*(tmp_samp_w/tmp_samp_w.sum()))
        sample_w = new_classes.copy().astype(float)
        sample_w[new_classes==0] = tmp_samp_w[0]
        sample_w[new_classes==1] = tmp_samp_w[1]
        sample_w[new_classes==2] = tmp_samp_w[2]
        sample_w[new_classes==3] = tmp_samp_w[3]
        '''
        #gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedKFold(hm_y,n_folds=gs),fit_params=dict(sample_weight=sample_w), n_jobs=-1,scoring='accuracy')
        #gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedKFold(hm_y,n_folds=gs),fit_params=dict(sample_weight=proba), n_jobs=-1,scoring='accuracy')
        gridclf = GridSearchCV(clf2, param_grid=param_grid, cv=StratifiedKFold(hm_y,n_folds=gs), n_jobs=-1,scoring='accuracy')
        gridclf.fit(xwl2,hm_y)
        clf2 = gridclf.best_estimator_
        #clf2.fit(xwl2,hm_y)
        print 'class order : ', clf2.classes_
        # train right classifier on easy cases
        #self.clf_easy = SVC(C=1.,cache_size=500,kernel='linear',class_weight='balanced',probability=False)
        #l2_pred = clf2.predict(xwl2)
        #self.clf_easy.fit(xw[l2_pred>0,:],y[l2_pred>0])

        if self.verbose:
            print clf2
            print clf2.coef_

        self.clf2 = clf2

    def predict(self,x,xwl2):
        xw = x.copy()#[:,self.w_select]
        y_pred1 = self.clf1.predict(xw)
        y_pred2 = self.clf2.decision_function(xwl2)
        #print self.clf2.predict_proba(xwl2)
        #y_pred2 = self.clf2.predict_proba(xwl2)[:,1]-0.6
        return np.array([y_pred1,y_pred2]).T

    def estimate_hitmiss(self,x,y):
        #return clf.predict(x)==y,clf.predict(x)

        # Perform a LOO to estimate the actual HM
        hm_results = []
        predictions =[]
        for i in range(len(y)):
            train_idx = np.array(np.hstack((np.arange(0,i),np.arange(i+1,len(y)))),dtype=int)
            self.clf1.fit(x[train_idx,:],y[train_idx])
            #print clf.predict(x[i,:]) == y[i]
            hm_results.append(float(self.clf1.predict(x[i,:].reshape(1,-1)) == y[i].reshape(1,-1)))
            predictions.append(self.clf1.predict(x[i,:].reshape(1,-1)))
            #hm_results.append(int((y[i] == label) & (clf.predict(x[i,:]) == y[i]) ))#   clf.predict(x[i,:]) == y[i]))

        predictions = np.array(predictions)
        hm_results = np.array(hm_results)
        self.clf1.fit(x,y)
        return hm_results, predictions[:,0]

    def suffle_hm(self,x,y,gamma=0.5,n_iter=50):
        hm_count = np.zeros_like(y).astype(float)
        hm = np.zeros_like(y).astype(float)
        skf = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=.25,random_state=np.random.seed(42))
        coefs_ = []
        intercepts_ = []
        for train,test in skf:
            self.clf1.fit(x[train,:],y[train])
            hm_count[test] += 1.
            hm[test] += (self.clf1.predict(x[test,:])==y[test]).astype(float)
            coefs_.append(self.clf1.coef_)
            intercepts_.append(self.clf1.intercept_)
        proba = hm/hm_count
        print hm_count
        print proba
        #self.clf1.coef_ = np.array(coefs_).mean(0)
        #self.clf1.intercept_ = np.array(intercepts_).mean(0)
        self.clf1.fit(x,y)
        #self.clf1.fit(x[proba>0.3,:],y[proba>0.3])
        return (proba>gamma).astype(int),proba

