__author__ = 'Christian Dansereau'

import numpy as np
from proteus.predic import prediction
from proteus.predic import subtypes
from proteus.predic.high_confidence import TwoStagesPrediction
from sklearn.svm import SVC, LinearSVC, l1_min_c
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import LeaveOneOut, LeavePOut, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
import time
from proteus.io import sbp_util


def compute_loo_parall((net_data_low_main, y, confounds, n_subtypes, train_index, test_index)):
    my_sbp = SBP()
    my_sbp.fit(net_data_low_main[train_index, ...], y[train_index], confounds[train_index, ...], n_subtypes,
               verbose=False)
    tmp_scores = my_sbp.predict(net_data_low_main[test_index, ...], confounds[test_index, ...])
    return np.hstack((y[test_index], tmp_scores[0][0], tmp_scores[1][0]))


class SBP:
    '''
    Pipeline for subtype base prediction
    '''

    def __init__(self, verbose=True, dynamic=True, stage1_model_type='svm', nSubtypes=7,
                 nSubtypes_stage2=0, mask_part=[], stage1_metric='accuracy', stage2_metric='f1_weighted',
                 s2_branches=True, min_gamma=0.8, thresh_ratio=0.1, n_iter=100, shuffle_test_split=0.2, gamma=1.,
                 gamma_auto_adjust=True, flag_recurrent=False, recurrent_modes=3):
        self.verbose = verbose
        self.dynamic = dynamic
        self.gamma = gamma
        self.min_gamma = min_gamma
        self.thresh_ratio = thresh_ratio
        self.mask_part = mask_part
        self.stage1_model_type = stage1_model_type
        self.nSubtypes = nSubtypes
        self.stage1_metric = stage1_metric
        self.stage2_metric = stage2_metric
        self.s2_branches = s2_branches
        self.n_iter = n_iter
        self.shuffle_test_split = shuffle_test_split
        self.gamma_auto_adjust = gamma_auto_adjust
        self.flag_recurrent = flag_recurrent
        self.recurrent_modes = recurrent_modes

        if nSubtypes_stage2 == 0:
            self.nSubtypes_stage2 = self.nSubtypes
        else:
            self.nSubtypes_stage2 = nSubtypes_stage2

    def get_w(self, x, confounds):
        ### extract w values
        W = []
        W2 = []
        for ii in range(len(self.st_crm)):
            W.append(self.st_crm[ii][1].compute_weights(self.st_crm[ii][0].transform(confounds, x[:, ii, :]),
                                                        mask_part=self.mask_part))
            W2.append(self.st_crm[ii][2].compute_weights(self.st_crm[ii][0].transform(confounds, x[:, ii, :]),
                                                         mask_part=self.mask_part))
        xw = np.hstack(W)
        xw2 = np.hstack(W2)
        return subtypes.reshapeW(xw), subtypes.reshapeW(xw2)

    def get_w_files(self, files_path, subjects_id_list, confounds):
        ### extract w values
        W = []
        W2 = []
        for ii in range(len(self.st_crm)):
            x_ref = sbp_util.grab_rmap(subjects_id_list, files_path, ii, dynamic=False)
            ## compute w values
            W.append(self.st_crm[ii][1].compute_weights(self.st_crm[ii][0].transform(confounds, x_ref),
                                                        mask_part=self.mask_part))

            W2.append(self.st_crm[ii][2].compute_weights(self.st_crm[ii][0].transform(confounds, x_ref),
                                                         mask_part=self.mask_part))
            del x_ref

        xw = np.hstack(W)
        xw2 = np.hstack(W2)
        return subtypes.reshapeW(xw), subtypes.reshapeW(xw2)

    def fit(self, x_dyn, confounds_dyn, x, confounds, y, extra_var=[]):

        if self.verbose: start = time.time()
        ### train subtypes
        self.st_crm = []
        for ii in range(x.shape[1]):
            crm = prediction.ConfoundsRm(confounds_dyn, x_dyn[:, ii, :])
            # st
            st = subtypes.clusteringST()
            st.fit_network(crm.transform(confounds_dyn, x_dyn[:, ii, :]), nSubtypes=self.nSubtypes)
            # stage 2                                                                                                                                         
            st_s2 = subtypes.clusteringST()
            st_s2.fit_network(crm.transform(confounds_dyn, x_dyn[:, ii, :]), nSubtypes=self.nSubtypes_stage2)
            self.st_crm.append([crm, st, st_s2])

        ### extract w values
        xw, xw2 = self.get_w(x, confounds)
        print('xw sub data', xw[0, :])
        if self.verbose: print("Subtype extraction, Time elapsed: {}s)".format(int(time.time() - start)))

        ### Include extra covariates
        if len(extra_var) != 0:
            all_var = np.hstack((xw, extra_var))
            all_var_s2 = np.hstack((xw2, extra_var))
        else:
            all_var = xw
            all_var_s2 = xw2

        ### prediction model
        if self.verbose: start = time.time()
        #self.tlp = TwoLevelsPrediction(self.verbose, stage1_model_type=self.stage1_model_type, gamma=self.gamma,
        #                               stage1_metric=self.stage1_metric, stage2_metric=self.stage2_metric,
        #                               s2_branches=self.s2_branches)
        self.tlp = TwoStagesPrediction(self.verbose, thresh_ratio=self.thresh_ratio, min_gamma=self.min_gamma, shuffle_test_split=self.shuffle_test_split, n_iter=self.n_iter, gamma_auto_adjust=self.gamma_auto_adjust, recurrent_modes=self.recurrent_modes)
        #self.tlp_recurrent = TwoStagesPrediction(self.verbose, thresh_ratio=self.thresh_ratio, min_gamma=self.min_gamma)
        self.tlp.fit(all_var, all_var_s2, y)
        # self.tlp_recurrent.fit_recurrent(all_var, all_var_s2, y)
        if self.verbose: print("Two Stages prediction, Time elapsed: {}s)".format(int(time.time() - start)))

    def fit_files_st(self, files_path_st, subjects_id_list_st, confounds_st, files_path, subjects_id_list, confounds, y,
                     n_seeds, extra_var=[]):
        '''
        Use a list of subject IDs and search for them in the path, grab the results per network.
        Same as fit_files() except that you can train and test on different set of data
        '''
        if self.verbose: start = time.time()
        ### train subtypes
        self.st_crm = []
        # for ii in [5,13]:#range(x.shape[1]):
        xw = []
        for ii in range(n_seeds):
            print('Train seed ' + str(ii + 1))
            if self.dynamic:
                [x_dyn, x_ref] = sbp_util.grab_rmap(subjects_id_list_st, files_path_st, ii, dynamic=self.dynamic)
                confounds_dyn = []
                for jj in range(len(x_dyn)):
                    confounds_dyn.append((confounds_st[jj],) * x_dyn[jj].shape[0])
                confounds_dyn = np.vstack(confounds_dyn)
                x_dyn = np.vstack(x_dyn)
            else:
                x_ref = sbp_util.grab_rmap(subjects_id_list_st, files_path_st, ii, dynamic=self.dynamic)
                x_dyn = x_ref
                confounds_dyn = confounds_st

            del x_ref
            ## regress confounds
            crm = prediction.ConfoundsRm(confounds_dyn, x_dyn)
            ## extract subtypes
            st = subtypes.clusteringST()
            st.fit_network(crm.transform(confounds_dyn, x_dyn), nSubtypes=self.nSubtypes)
            # stage 2
            st_s2 = subtypes.clusteringST()
            st_s2.fit_network(crm.transform(confounds_dyn, x_dyn), nSubtypes=self.nSubtypes_stage2)
            self.st_crm.append([crm, st, st_s2])
            del x_dyn

        if self.verbose: print("Subtype extraction, Time elapsed: {}s)".format(int(time.time() - start)))


    def fit_files(self, files_path, subjects_id_list, confounds, y, n_seeds, extra_var=[], skip_st_training=False):
        '''
        use a list of subject IDs and search for them in the path, grab the results per network
        '''
        if skip_st_training == False:
            self.fit_files_st(files_path, subjects_id_list, confounds, files_path, subjects_id_list, confounds, y, n_seeds,
                              extra_var)

        # compute the W
        xw, xw2 = self.get_w_files(files_path, subjects_id_list, confounds)

        ### Include extra covariates
        if len(extra_var) != 0:
            all_var = np.hstack((xw, extra_var))
            all_var_s2 = np.hstack((xw2, extra_var))
        else:
            all_var = xw
            all_var_s2 = xw2

        ### prediction model
        if self.verbose: start = time.time()
        # self.tlp = TwoLevelsPrediction(self.verbose, stage1_model_type=self.stage1_model_type, gamma=self.gamma,
        #                               stage1_metric=self.stage1_metric, stage2_metric=self.stage2_metric)
        self.tlp = TwoStagesPrediction(self.verbose, thresh_ratio=self.thresh_ratio, min_gamma=self.min_gamma, shuffle_test_split=self.shuffle_test_split, n_iter=self.n_iter, gamma_auto_adjust=self.gamma_auto_adjust, recurrent_modes=self.recurrent_modes)
        # self.tlp_recurrent = TwoStagesPrediction(self.verbose, thresh_ratio=self.thresh_ratio, min_gamma=self.min_gamma)
        if self.flag_recurrent:
            self.tlp.fit_recurrent(all_var, all_var_s2, y)
        else:
            self.tlp.fit(all_var, all_var_s2, y)

        if self.verbose: print("Two Levels prediction, Time elapsed: {}s)".format(int(time.time() - start)))

    def predict_files(self, files_path, subjects_id_list, confounds, extra_var=[], recurrent=False):
        xw, xw2 = self.get_w_files(files_path, subjects_id_list, confounds)
        return self.predict([xw, xw2], [], extra_var, skip_confounds=True, recurrent=recurrent)

    def predict(self, x, confounds, extra_var=[], skip_confounds=False, recurrent=False):

        if skip_confounds:
            xw = x[0]
            xw2 = x[1]
        else:
            xw, xw2 = self.get_w(x, confounds)

        ### Include extra covariates
        if len(extra_var) != 0:
            all_var = np.hstack((xw, extra_var))
            all_var_s2 = np.hstack((xw2, extra_var))
        else:
            all_var = xw
            all_var_s2 = xw2

        ### prediction model
        #if recurrent:
            # return self.tlp_recurrent.predict(all_var, all_var_s2)
        #else:
        data_array, dict_array = self.tlp.predict(all_var, all_var_s2)
        return data_array

    def _score(self, y, res):
        l1_y_pred = (res[:, 0] > 0).astype(int)
        risk_mask = res[:, 1] > 0
        right_cases = accuracy_score(y[risk_mask], l1_y_pred[risk_mask])
        left_cases = accuracy_score(y[~risk_mask], l1_y_pred[~risk_mask])
        self.res = np.hstack((y[:, np.newaxis], res))
        self.scores = (accuracy_score(y, l1_y_pred), left_cases, right_cases)

    def score_files(self, files_path, subjects_id_list, confounds, y, extra_var=[], recurrent=False):
        res = self.predict_files(files_path, subjects_id_list, confounds, extra_var, recurrent=recurrent)
        self._score(y, res)
        return self.scores

    def score(self, x, confounds, y, extra_var=[], recurrent=False):
        res = self.predict(x, confounds, extra_var, recurrent=recurrent)
        self._score(y, res)
        return self.scores

    def estimate_acc(self, net_data_low_main, y, confounds, n_subtypes, verbose=False):

        sss = LeaveOneOut(len(y))
        # scores: y, y_pred, decision_function
        self.scores = []
        k = 0
        for train_index, test_index in sss:
            k += 1
            print('Fold: ' + str(k) + '/' + str(len(y)))
            self.fit(net_data_low_main[train_index, ...], y[train_index], confounds[train_index, ...],
                     n_subtypes=n_subtypes, verbose=False, flag_feature_select=False)
            tmp_scores = self.predict(net_data_low_main[test_index, ...], confounds[test_index, ...])
            self.scores.append(np.hstack((y[test_index], tmp_scores[0][0], tmp_scores[0][1])))
        self.scores = np.array(self.scores)

    def estimate_acc_multicore(self, net_data_low_main, y, confounds, n_subtypes, verbose=False):
        taskList_loo = []
        sss = LeaveOneOut(len(y))
        # scores: y, y_pred, decision_function
        self.scores = []
        k = 0
        for train_index, test_index in sss:
            taskList_loo.append((net_data_low_main, y, confounds, n_subtypes, train_index, test_index))

        pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 2))  # Don't use all my processing power.
        r2 = pool.map_async(compute_loo_parall, taskList_loo,
                            callback=self.scores.append)  # Using fxn "calculate", feed taskList, and values stored in "results" list
        r2.wait()
        pool.terminate()
        pool.join()
        self.scores = np.array(self.scores)


class nullClassifier():
    def fit(self, x, y):
        pass

    def predict(self, x):
        return np.zeros_like(x[:, 0])

    def decision_function(self, x):
        return -np.ones_like(x[:, 0])

'''
class TwoLevelsPrediction:

    #2 Level prediction


    def __init__(self, verbose=True, stage1_model_type='svm', gamma=0.999, stage1_metric='accuracy',
                 stage2_metric='f1_weighted', s2_branches=True):
        self.verbose = verbose
        self.stage1_model_type = stage1_model_type
        self.gamma = gamma
        self.stage1_metric = stage1_metric
        self.stage2_metric = stage2_metric
        self.s2_branches = s2_branches

    def adjust_gamma(self, proba, thresh=0.1):
        gamma = self.gamma
        while (np.mean(proba > gamma) <= thresh) and (gamma > 0.8):
            gamma = gamma - 0.01
        if (np.mean(proba > gamma) <= thresh):
            return np.zeros_like(proba), gamma

        return (proba > gamma).astype(int), gamma

    def fit(self, xw, xwl2, y, gs=4, retrain_l1=False):

        print 'Stage 1'
        if self.stage1_model_type == 'logit':
            clf = LogisticRegression(C=1, class_weight='balanced', penalty='l2', max_iter=300)
        elif self.stage1_model_type == 'svm':
            clf = SVC(C=1., cache_size=500, kernel='linear', class_weight='balanced', probability=False)
        elif self.stage1_model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=20, class_weight='balanced')

        # Stage 1
        if self.stage1_model_type == 'logit':
            param_grid = dict(C=(5, 5.0001))
        elif self.stage1_model_type == 'svm':
            param_grid = dict(C=(np.logspace(-2, 1, 15)))
        elif self.stage1_model_type == 'rf':
            param_grid = dict(n_estimators=(20, 10))

        gridclf = GridSearchCV(clf, param_grid=param_grid,
                               cv=StratifiedShuffleSplit(n_splits=50, test_size=.2, random_state=1), n_jobs=-1,
                               scoring=self.stage1_metric)
        gridclf.fit(xw, y)
        self.clf1 = gridclf.best_estimator_

        # self.clf1 = clf.fit(xw,y)
        if self.verbose:
            print self.clf1
            # print self.clf1.coef_
        # hm_y,y_pred_train = self.estimate_hitmiss(xw,y)
        hm_y, proba = self.suffle_hm(xw, y, gamma=self.gamma, n_iter=100)
        hm_y, auto_gamma = self.adjust_gamma(proba)
        self.auto_gamma = auto_gamma

        if self.verbose: proba
        if self.verbose: print 'Average hm score', np.mean(hm_y)
        # print 'n stage3 ',(proba>gamma).sum()
        # self.clf3.fit(xw[proba>gamma,:],y[proba>gamma])
        # if retrain_l1:
        #    self.clf1 = self.clf3
        print 'Stage 2'
        # Stage 2
        min_c = l1_min_c(xwl2, hm_y, loss='log')
        clf2 = LogisticRegression(C=1., class_weight='balanced', penalty='l1', solver='liblinear', max_iter=300)

        # if min_c>(10**-0.2):
        #    param_grid = dict(C=(np.logspace(np.log10(min_c), 1, 15)))
        # else:
        param_grid = dict(C=(np.logspace(-.2, 1, 15)))

        # 2 levels balancing 

        gridclf = GridSearchCV(clf2, param_grid=param_grid,
                               cv=StratifiedShuffleSplit(n_splits=50, test_size=.2, random_state=1), n_jobs=-1,
                               scoring=self.stage2_metric)

        gridclf.fit(xwl2, hm_y)
        clf2 = gridclf.best_estimator_
        if self.verbose:
            print clf2
            print clf2.coef_

        self.clf2 = clf2

        if self.s2_branches:
            self.fit_2branch(xwl2, proba, y)
            # self.robust_coef(xwl2,hm_y)

    def fit_branchmodel(self, xwl2, hm_y):
        clf = LogisticRegression(C=1., class_weight='balanced', penalty='l1', solver='liblinear', max_iter=300)
        param_grid = dict(C=(np.logspace(-.2, 1, 15)))
        gridclf = GridSearchCV(clf, param_grid=param_grid,
                               cv=StratifiedShuffleSplit(n_splits=50, test_size=.2, random_state=1), n_jobs=-1,
                               scoring=self.stage2_metric)
        # train
        if len(np.unique(hm_y)) > 1:
            gridclf.fit(xwl2, hm_y)
            return gridclf.best_estimator_
        else:
            return nullClassifier()

    def fit_2branch(self, xwl2, proba, y_pred):
        mask_ = y_pred == 0
        hm_tmp = proba.copy()
        hm_tmp[~mask_] = 0
        hm_y, auto_gamma = self.adjust_gamma(hm_tmp)
        self.clf_0 = self.fit_branchmodel(xwl2, hm_y)
        hm_tmp = proba.copy()
        hm_tmp[mask_] = 0
        hm_y, auto_gamma = self.adjust_gamma(hm_tmp)
        print proba, hm_tmp, hm_y, auto_gamma
        self.clf_1 = self.fit_branchmodel(xwl2, hm_y)

    def predict_2branch(self, xwl2):
        dfs2_cls0 = self.clf_0.decision_function(xwl2)
        dfs2_cls1 = self.clf_1.decision_function(xwl2)

        # unified decision XOR
        xor_mask = np.logical_xor(dfs2_cls0 > 0, dfs2_cls1 > 0)
        or_mask = np.logical_or(dfs2_cls0 > 0, dfs2_cls1 > 0)
        df_ = dfs2_cls0.copy()
        df_[dfs2_cls1 > 0] = dfs2_cls1[dfs2_cls1 > 0]
        df_[or_mask][~xor_mask[or_mask]] = -df_[or_mask][~xor_mask[or_mask]]

        return df_, dfs2_cls0, dfs2_cls1

    def robust_coef(self, xwl2, hm_y, n_iter=100):
        skf = StratifiedShuffleSplit(n_splits=n_iter, test_size=.2, random_state=1)
        coefs_ = []
        intercept_ = []
        for train, test in skf.split(xwl2, hm_y):
            self.clf2.fit(xwl2[train, :], hm_y[train])
            coefs_.append(self.clf2.coef_)
            intercept_.append(self.clf2.intercept_)
        self.clf2.coef_ = np.stack(coefs_).mean(0)
        self.clf2.intercept_ = np.stack(intercept_).mean(0)

    def predict(self, xw, xwl2):
        y_pred1 = self.clf1.decision_function(xw)
        if self.s2_branches:
            y_pred2_merge, hclr, hchr = self.predict_2branch(xwl2)
            y_pred2 = self.clf2.decision_function(xwl2)
            return np.array([y_pred1, y_pred2_merge, hclr, hchr, y_pred2]).T
        else:
            y_pred2 = self.clf2.decision_function(xwl2)
            return np.array([y_pred1, y_pred2]).T

    def score(self, xw, xwl2, y):
        res = self.predict(xw, xwl2)
        l1_y_pred = (res[:, 0] > 0).astype(int)
        risk_mask = res[:, 1] > 0
        right_cases = accuracy_score(y[risk_mask], l1_y_pred[risk_mask])
        left_cases = accuracy_score(y[~risk_mask], l1_y_pred[~risk_mask])

        # print 'clf3: ',self.clf3.score(xw,y)
        return accuracy_score(y, l1_y_pred), left_cases, right_cases

    def suffle_hm(self, x, y, gamma=0.5, n_iter=100):
        hm_count = np.zeros_like(y).astype(float)
        hm = np.zeros_like(y).astype(float)
        skf = StratifiedShuffleSplit(n_splits=n_iter, test_size=.2, random_state=1)
        coefs_ = []
        sv_ = []
        for train, test in skf.split(x, y):
            self.clf1.fit(x[train, :], y[train])
            hm_count[test] += 1.
            hm[test] += (self.clf1.predict(x[test, :]) == y[test]).astype(float)
            # coefs_.append(self.clf1.dual_coef_)
            # coefs_.append(self.clf1.coef_)
            # sv_.append(self.clf1.support_vectors_)
        proba = hm / hm_count
        if self.verbose:
            print(hm_count)
            print(proba)
        # self.clf1.dual_coef_ = np.stack(coefs_).mean(0)
        # self.clf1.support_vectors_ = np.stack(sv_).mean(0)
        # self.clf1.coef_ = np.stack(coefs_).mean(0)
        self.clf1.fit(x, y)
        return (proba >= gamma).astype(int), proba
'''