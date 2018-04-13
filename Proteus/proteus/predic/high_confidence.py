__author__ = 'Christian Dansereau'

"""
Highly predictable cases framework
This is a generic implementation of the CPF
Copyright 2016 Christian Dansereau all right reserved
"""
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster


class nullClassifier(object):
    def fit(self, x, y):
        pass

    def predict(self, x):
        return np.zeros_like(x[:, 0])

    def decision_function(self, x):
        return -np.ones_like(x[:, 0])


class BaseSvc(object):
    def __init__(self, scoring_metric='accuracy', param_grid=dict(C=(np.logspace(-2, 1, 15)))):
        self.scoring_metric = scoring_metric
        self.param_grid = param_grid
        self.clf = SVC(C=1., cache_size=1000, kernel='linear', class_weight='balanced', probability=False,
                       decision_function_shape='ovr')
        self.gridclf = GridSearchCV(self.clf, param_grid=self.param_grid,
                                    cv=StratifiedShuffleSplit(n_splits=50, test_size=.2, random_state=1), n_jobs=-1,
                                    scoring=self.scoring_metric)
        #self.gridclf = GridSearchCV(self.clf, param_grid=self.param_grid,
        #                            cv=5, n_jobs=-1,
        #                           scoring=self.scoring_metric)


    def fit(self, x, y, hyperparams_optim=True):
        if hyperparams_optim & (np.sum(y) > 2):

            self.gridclf.fit(x, y)
            self.clf = self.gridclf.best_estimator_
            self.gridclf.cv_results_ = None
        else:
            self.clf.fit(x, y)

    def predict(self, x):
        if getattr(self, 'clf', None) is None:
            print('The model was not fit before prediction')
            return None
        return self.clf.predict(x)

    def decision_function(self, x):
        if getattr(self, 'clf', None) is None:
            print('The model was not fit before prediction')
            return None
        return self.clf.decision_function(x)


class BaseLR(object):
    def __init__(self, scoring_metric='accuracy', param_grid=dict(C=(np.logspace(-0.2, 1, 15)))):
        self.scoring_metric = scoring_metric
        self.param_grid = param_grid
        clf = LogisticRegression(C=1., class_weight='balanced', penalty='l2', solver='liblinear', max_iter=300)
        self.gridclf = GridSearchCV(clf, param_grid=self.param_grid,
                                    cv=StratifiedShuffleSplit(n_splits=50, test_size=.2, random_state=1), n_jobs=-1,
                                    scoring=self.scoring_metric)

    def fit(self, x, y, hyperparams_optim=True):
        if hyperparams_optim:
            self.gridclf.fit(x, y)
            self.clf = self.gridclf.best_estimator_
            self.gridclf.cv_results_ = None
        else:
            self.clf.fit(x, y)

    def predict(self, x):
        if getattr(self, 'clf', None) is None:
            print('The model was not fit before prediction')
            return None
        return self.clf.predict(x)

    def decision_function(self, x):
        if getattr(self, 'clf', None) is None:
            print('The model was not fit before prediction')
            return None
        return self.clf.decision_function(x)


class ConfidenceLR(object):
    def __init__(self, scoring_metric='f1_weighted', param_grid=dict(C=(np.logspace(-0.2, 1, 15)))):
        self.scoring_metric = scoring_metric
        self.param_grid = param_grid
        self.t = []

    def _fit_branchmodel(self, xwl2, hm_y):
        clf = LogisticRegression(C=1., class_weight='balanced', penalty='l1', solver='liblinear', max_iter=300)
        gridclf = GridSearchCV(clf, param_grid=self.param_grid,
                               cv=StratifiedShuffleSplit(n_splits=50, test_size=.2, random_state=1), n_jobs=-1,
                               scoring=self.scoring_metric)
        # train
        if (len(np.unique(hm_y)) > 1) & (np.sum(hm_y) >= 2):
            gridclf.fit(xwl2, hm_y)
            return gridclf.best_estimator_
        else:
            return nullClassifier()

    def fit(self, x, hm_1hot):
        self.clfs = []
        # k = 0
        for hm in hm_1hot:
            self.clfs.append(self._fit_branchmodel(x, hm))
            # hm_proba = self.clfs[-1].predict_proba(x)[:, k]

            # t = 0.5
            # while np.mean(hm[hm_proba > t]) < 1:
            #    t += 0.001
            # self.t.append(t)
            # k += 1

    def decision_function(self, x):
        if getattr(self, 'clfs', None) == None:
            print('The model was not fit before prediction')
            return None
        df = []
        # k = 0
        for clf in self.clfs:
            tmp_df = np.array(clf.decision_function(x))
            # hm_proba = self.clfs[k].predict_proba(x)[:, k]
            # tmp_df[hm_proba > self.t[k]] = -tmp_df[hm_proba > self.t[k]]
            df.append(tmp_df)
            # k += 1

        return np.stack(df).T


class MulticlassLR(object):
    def __init__(self, scoring_metric='f1_weighted', param_grid=dict(C=(np.logspace(-0.2, 1, 15)))):
        self.scoring_metric = scoring_metric
        self.param_grid = param_grid

    def _fit_branchmodel(self, xwl2, hm_y):
        clf = LogisticRegression(C=1., class_weight='balanced', penalty='l1', solver='liblinear', max_iter=300)
        gridclf = GridSearchCV(clf, param_grid=self.param_grid,
                               cv=StratifiedShuffleSplit(n_splits=50, test_size=.2, random_state=1), n_jobs=-1,
                               scoring=self.scoring_metric)
        # train
        if len(np.unique(hm_y)) > 1:
            gridclf.fit(xwl2, hm_y)
            return gridclf.best_estimator_
        else:
            return nullClassifier()

    def fit(self, x, hm_1hot):
        self.clfs = []
        for hm in hm_1hot:
            self.clfs.append(self._fit_branchmodel(x, hm))

    def decision_function(self, x):
        if getattr(self, 'clfs', None) == None:
            print('The model was not fit before prediction')
            return None
        df = []
        for clf in self.clfs:
            df.append(clf.decision_function(x))

        return np.stack(df).T


class HC_LR(object):
    def __init__(self, scoring_metric='accuracy', param_grid=dict(C=(np.logspace(-0.2, 1, 15)))):
        self.scoring_metric = scoring_metric
        self.param_grid = param_grid
        clf = LogisticRegression(C=1., class_weight='balanced', penalty='l1', solver='liblinear', max_iter=300)
        self.gridclf = GridSearchCV(clf, param_grid=self.param_grid,
                                    cv=StratifiedShuffleSplit(n_splits=50, test_size=.2, random_state=1), n_jobs=-1,
                                    scoring=self.scoring_metric)

    def fit(self, x, y):
        self.gridclf.fit(x, y)
        self.clf = self.gridclf.best_estimator_
        self.gridclf.cv_results_ = None

    def predict(self, x):
        if getattr(self, 'clf', None) is None:
            print('The model was not fit before prediction')
            return None
        return self.clf.predict(x)

    def decision_function(self, x):
        if getattr(self, 'clf', None) is None:
            print('The model was not fit before prediction')
            return None
        return self.clf.decision_function(x)

    def predict_proba(self, x):
        if getattr(self, 'clf', None) is None:
            print('The model was not fit before prediction')
            return None
        return self.clf.predict_proba(x)


class HitProbability(object):
    def __init__(self, scoring_metric='r2', param_grid=dict(C=(np.logspace(-0.1, 0.1, 15)))):
        self.scoring_metric = scoring_metric
        self.param_grid = param_grid
        clf = SVR(C=1., cache_size=500, kernel='linear')
        self.gridclf = GridSearchCV(clf, param_grid=self.param_grid,
                                    cv=ShuffleSplit(n_splits=50, test_size=.2, random_state=1), n_jobs=-1,
                                    scoring=self.scoring_metric)

    def fit(self, x, y):
        #self.gridclf.fit(x, y)
        #self.clf = self.gridclf.best_estimator_
        #self.gridclf.cv_results_ = None

        self.clf = nullClassifier()

    def predict(self, x):
        if getattr(self, 'clf', None) is None:
            print('The model was not fit before prediction')
            return None
        return self.clf.predict(x)


class TwoStagesPrediction(object):
    """
    Two Stage prediction framework
    """

    def __init__(self, verbose=True, basemodel=[], confidencemodel=[], gamma=1., n_iter=100, min_gamma=0.8,
                 thresh_ratio=0.1, shuffle_test_split=0.2, gamma_auto_adjust=True, recurrent_modes=3, hitprobability_strategy='shuffle'):
        self.verbose = verbose
        self.gamma = gamma
        self.n_iter = n_iter
        self.scaler_s1 = StandardScaler(with_mean=True, with_std=False)
        self.scaler_s2 = StandardScaler(with_mean=True, with_std=False)
        self.min_gamma = min_gamma
        self.thresh_ratio = thresh_ratio
        self.shuffle_test_split = shuffle_test_split
        self.gamma_auto_adjust = gamma_auto_adjust
        self.recurrent_modes = recurrent_modes
        self.hitprobability_strategy = hitprobability_strategy

        if basemodel == []:
            self.basemodel = BaseSvc()
        else:
            self.basemodel = basemodel
        if confidencemodel == []:
            self.confidencemodel = ConfidenceLR()
        else:
            self.confidencemodel = confidencemodel

    def _adjust_gamma(self, proba):

        thresh = self.thresh_ratio
        min_gamma = self.min_gamma
        gamma_auto_adjust = self.gamma_auto_adjust
        gamma = self.gamma

        if gamma_auto_adjust:
            while (np.mean(proba >= gamma) <= thresh) and (gamma > min_gamma):
                gamma = gamma - 0.01

        # if (np.mean(proba > gamma) <= thresh):
        #    return np.zeros_like(proba), gamma

        return (proba >= gamma).astype(int), gamma

    def _adjust_gamma_classes(self, proba, y):

        hm_y = np.zeros_like(y).astype(float)
        auto_gamma = []
        for y_class in np.unique(y):
            mask_ = y == y_class
            hm_, adjusted_gamma = self._adjust_gamma(proba[mask_])
            hm_y[mask_] = hm_
            auto_gamma.append(adjusted_gamma)

        return hm_y, auto_gamma

    def fit(self, x, x2, y):
        """
        Fit the Two stage model on the data x and x2
        :param x: Input matrix of examples X features for stage 1
        :param x2: Input matrix of examples X features for stage 2
        :param y: Target labels
        :return
        """
        print('Stage 1')
        x_ = self.scaler_s1.fit_transform(x)
        self.basemodel.fit(x_, y)
        self.training_hit_probability = self._hitprobability(x_, y)

        # Learn the hit probability
        self.hitproba = HitProbability()
        self.hitproba.fit(x_, self.training_hit_probability)

        # Learn high confidence for all classes
        hm_y, auto_gamma = self._adjust_gamma_classes(self.training_hit_probability, y)
        self.joint_class_hc = HC_LR()
        self.joint_class_hc.fit(x_, hm_y)

        if self.verbose:
            print('Average hm score', str(np.mean(hm_y)))

        print('Stage 2')
        # Stage 2
        hm_1hot = self._one_hot(self.training_hit_probability, y)

        # Train stage2
        x2_ = self.scaler_s2.fit_transform(x2)
        self.confidencemodel.fit(x2_, hm_1hot)

    def fit_recurrent(self, x, x2, y):
        """
        Fit the Two stage model on the data x and x2
        Note that this training strategy focus only on the class 1
        :param x: Input matrix of examples X features for stage 1
        :param x2: Input matrix of examples X features for stage 2
        :param y: Target labels
        :return
        """
        # print('Stage 1')
        x_ = self.scaler_s1.fit_transform(x)
        x2_ = self.scaler_s2.fit_transform(x2)

        self.basemodel.fit(x_, y)
        self.training_hit_probability = self._hitprobability(x_, y)

        # Learn the hit probability
        self.hitproba = HitProbability()
        self.hitproba.fit(x_, self.training_hit_probability)

        # Learn high confidence for all classes
        hm_y, auto_gamma = self._adjust_gamma(self.training_hit_probability)
        self.joint_class_hc = HC_LR()
        self.joint_class_hc.fit(x_, hm_y)

        # hm_subtypes = []
        # proba_subtypes = []

        # while np.mean(y_) > 0.01:
        # for label in np.unique(y):

        hm_1hot = []
        hm_1hot.append(self._one_hot(self.training_hit_probability, y)[0])
        y_ = y.copy()

        self.recurrent_base = []
        self.recurrent_hpc = []
        for ii in range(self.recurrent_modes):
            print('Stage 1 iter: ' + str(ii))
            #self.recurrent_base.append(BaseSvc())

            if np.sum(y_) > 2:
                self.basemodel = BaseSvc()
                hm_y, proba_tmp = self._fit_mode(x_, y_)
                hm_candidate = self._one_hot(proba_tmp, y_)[1]
            else:
                hm_candidate = np.zeros_like(y_)

            self.recurrent_base.append(self.basemodel)

            #if np.sum(hm_candidate) >= 2:
            hm_1hot.append(hm_candidate)

            # remove the selected subgroup from the target list
            y_[hm_1hot[-1] == 1] = 0

        # make the default base model the first
        self.basemodel = self.recurrent_base[0]

        print('Stage 2')
        # Stage 2
        # hm_1hot = hm_subtypes
        # train stage2
        x2_ = self.scaler_s2.fit_transform(x2)
        self.confidencemodel.fit(x2_, hm_1hot)

    def _fit_mode(self, x, y):
        self.basemodel.fit(x, y)
        proba = self._hitprobability(x, y)
        mask_ = y == 1
        proba_tmp = proba.copy()
        proba_tmp[~mask_] = 0
        hm_y, auto_gamma = self._adjust_gamma(proba_tmp)  # , thresh=0.1, min_gamma=0.4)
        proba_tmp[hm_y != 1] = 0

        return hm_y, proba_tmp

    def _one_hot(self, proba, y):
        """
        One hot encoder create a binary vector for each class
        :param proba: probability of a hit from the base model
        :param y: labels for each observation
        :return: matrix observation X classes
        """
        hm_1hot = []
        for label in np.unique(y):
            mask_ = y == label
            proba_tmp = proba.copy()
            proba_tmp[~mask_] = 0
            hm_y, auto_gamma = self._adjust_gamma(proba_tmp)

            if self.verbose:
                print('Adjusted gamma: ', str(auto_gamma))
            hm_1hot.append(hm_y)
        return hm_1hot

    def predict(self, x, x2):
        """
        Predict labels for the given examples
        :param x: Input matrix of examples X features for stage 1
        :param x2: Input matrix of examples X features for stage 2
        :return: examples X [labels_stage1, merge_confidence_decision, decision_function_class0, decision_function_class1, ...]
        """

        x_ = self.scaler_s1.transform(x)
        x2_ = self.scaler_s2.transform(x2)
        y_df1 = self.basemodel.decision_function(x_)
        dfs = self.confidencemodel.decision_function(x2_)

        '''
        hc_df = np.max(dfs, 1)

        # high confidence vector
        hc_df[(dfs > 0).sum(1) > 1] = np.min(dfs, 1)[(dfs > 0).sum(1) > 1]
        '''

        hc_df = -np.ones((dfs.shape[0], 1))
        y_pred1 = self.basemodel.predict(x_)

        unique_labels = range(dfs.shape[1])
        for label in unique_labels:
            hc_df[y_pred1 == label, :] = dfs[y_pred1 == label, label][:, np.newaxis]

        '''
        # TODO modif this to be compatible with multilabel classifiers (currently only binary classifier)
        xor_mask = np.logical_xor(dfs[:, 0] > 0, dfs[:, 1] > 0)

        if xor_mask.sum() > 0:
            hc_df[xor_mask, 0] = np.max(dfs, 1)[xor_mask]
        hc_df[~xor_mask, 0] = -np.abs(np.max(dfs, 1)[~xor_mask])
        #hc_labels = np.greater(dfs[:, 1], dfs[:, 0]).astype(int)
        '''

        hit_proba_estimate = self.hitproba.predict(x_)
        joint_hc = self.joint_class_hc.predict(x_)

        data_array = []
        dict_array = []
        if len(joint_hc.shape):

            dict_array = {'s1df': y_df1[:, np.newaxis],
                          'hcdf': hc_df,
                          's2df': dfs,
                          'hitproba': hit_proba_estimate[:, np.newaxis],
                          'hcjoint': joint_hc[:, np.newaxis]
                          }
            data_array = np.hstack(
                # [y_df1[:, np.newaxis], hc_df, dfs, hit_proba_estimate[:, np.newaxis], joint_hc[:, np.newaxis]])
                [y_df1[:, np.newaxis], hc_df, dfs, hit_proba_estimate[:, np.newaxis], joint_hc[:, np.newaxis]])
            # [y_df1[:, np.newaxis], hc_labels[:, np.newaxis], hc_df, dfs, hit_proba_estimate[:, np.newaxis], joint_hc[:, np.newaxis]])
        else:
            # multiclass
            data_array = np.hstack([y_df1, hc_df, dfs, hit_proba_estimate, joint_hc])
            dict_array = {'s1df': y_df1,
                          'hcdf': hc_df,
                          's2df': dfs,
                          'hitproba': hit_proba_estimate,
                          'hcjoint': joint_hc
                          }

        return data_array, dict_array

    def _hitprobability(self, x, y):
        if self.hitprobability_strategy == 'clustering':
            return self._cluster_hitprobability(x, y)
        elif self.hitprobability_strategy == 'window':
            return self._window_hitprobability(x, y)
        elif self.hitprobability_strategy == 'avg_cluster':
            return self._avg_cluster_hitprobability(x, y)
        else:
            return self._shufflesplit(x, y)

    def _shufflesplit(self, x, y):
        """
        Random sampling to estimate the probability of a hit for each subjects from the parametrized model
        :param x: Input examples X features
        :param y: Labels to predict
        :return: Probability of hit for each examples
        """
        hm_count = np.zeros_like(y).astype(float)
        hm = np.zeros_like(y).astype(float)
        skf = StratifiedShuffleSplit(n_splits=self.n_iter, test_size=self.shuffle_test_split, random_state=1)

        for train, test in skf.split(x, y):
            # rnd_proba = np.abs(np.random.randn(len(train))*5.+1.)
            # rnd_proba[train==0] = 1.
            rnd_proba = None
            self.basemodel.fit(x[train, :], y[train], hyperparams_optim=False)
            hm_count[test] += 1.
            hm[test] += (self.basemodel.predict(x[test, :]) == y[test]).astype(float)

        proba = hm / hm_count
        if self.verbose:
            # print('H/M count:')
            # print(hm_count)
            print('Proba:')
            print(proba)
        self.basemodel.fit(x, y, hyperparams_optim=False)
        return proba

    def _avg_cluster_hitprobability(self, x, y, n_clusters=30):
        """
        Random sampling to estimate the probability of a hit for each subjects from the parametrized model
        :param x: Input examples X features
        :param y: Labels to predict
        :param n_clusters: Number of clusters, the individual hit probability inside each cluster will be averaged
        :return: Probability of hit for each examples
        """

        # Compute the individual Hit probability
        proba = self._shufflesplit(x, y)

        # average the individual hit probability for each cluster
        ind = self._cluster(x, x.shape[0]/2.)

        avg_proba = np.copy(proba)

        for cluster in np.unique(ind):
            mask_ = ind == cluster
            avg_proba[mask_] = avg_proba[mask_].mean()

        return avg_proba

    def _cluster_hitprobability(self, x, y):
        """
        Evaluate clusters to estimate the probability of a hit for each subjects from the parametrized model
        :param x: Input examples X features
        :param y: Labels to predict
        :return: Probability of hit for each examples
        """
        hm_count = np.zeros_like(y).astype(float)
        hm = np.zeros_like(y).astype(float)
        #skf = StratifiedShuffleSplit(n_splits=self.n_iter, test_size=self.shuffle_test_split, random_state=1)

        ind = self._cluster(x, 35)

        for cluster_id in np.unique(ind):
            test = np.argwhere(ind == cluster_id)[:, 0]
            train = np.argwhere(ind != cluster_id)[:, 0]
            #print test
            self.basemodel.fit(x[train, :], y[train], hyperparams_optim=False)
            hm_count[test] += 1.
            hm[test] += (self.basemodel.predict(x[test, :]) == y[test]).astype(float)

        proba = hm / hm_count
        if self.verbose:
            # print('H/M count:')
            # print(hm_count)
            print('Proba:')
            print(proba)
        self.basemodel.fit(x, y, hyperparams_optim=False)
        return proba

    def _window_hitprobability(self, x, y):
        """
        Evaluate a window of clusters to estimate the probability of a hit for each subjects from the parametrized model
        :param x: Input examples X features
        :param y: Labels to predict
        :return: Probability of hit for each examples
        """
        hm_count = np.zeros_like(y).astype(float)
        hm = np.zeros_like(y).astype(float)
        #skf = StratifiedShuffleSplit(n_splits=self.n_iter, test_size=self.shuffle_test_split, random_state=1)
        skf = StratifiedShuffleSplit(n_splits=self.n_iter, test_size=self.shuffle_test_split, random_state=1)

        ind = self._cluster(x, x.shape[0])
        ind = np.argsort(ind)

        for i in range(self.n_iter):
            # variable window size between 10% and 50%
            window_size = np.random.randint(len(ind)*0.1, len(ind)*0.5)
            train, test = self._window_indexes(ind, window_size)
            self.basemodel.fit(x[train, :], y[train], hyperparams_optim=False)
            hm_count[test] += 1.
            hm[test] += (self.basemodel.predict(x[test, :]) == y[test]).astype(float)

        proba = hm / hm_count
        if self.verbose:
            # print('H/M count:')
            # print(hm_count)
            print('Proba:')
            print(proba)
        self.basemodel.fit(x, y, hyperparams_optim=False)
        return proba

    def _cluster(self, x, n_clusters):
        # Normalize feature wise
        x_ = scale(x, axis=0, with_mean=True, with_std=True, copy=True)

        row_dist = np.corrcoef(x_)
        # row_dist = data
        row_clusters = linkage(row_dist, method='ward')
        ind = fcluster(row_clusters, n_clusters, criterion='maxclust')
        return ind

    def _window_indexes(self, x, test_size):
        seed_ = np.random.randint(0, x.shape[0] + 1)
        mask = np.zeros_like(x).astype(bool)

        if seed_ > x.shape[0] - test_size:
            n_pre_index = test_size - (x.shape[0] - seed_)
            mask[seed_ - n_pre_index:] = True
        else:
            mask[seed_:seed_ + test_size] = True
        train = x[~mask]
        test = x[mask]

        return train, test