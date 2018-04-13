__author__ = 'Christian Dansereau'

import numpy as np
from sklearn.feature_selection import SelectFpr
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from proteus.predic import predlib as plib
from sklearn import preprocessing
from proteus.matrix import tseries as ts
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import linear_model


def custom_scale(x):
    # print x
    # print np.mean(np.array(x),axis=0), np.std(np.array(x),axis=0)
    return (np.array(x) - np.array(x).mean(axis=0)) / np.array(x).std(axis=0)


def estimate_std(p, n):
    print(p, n, np.sqrt(p * (1 - p) / n))
    return np.sqrt(p * (1 - p) / n)


def estimate_unbalanced_std(y1, y2):
    if type(y1).__module__ != np.__name__:
        y1 = np.array(y1)
    if type(y2).__module__ != np.__name__:
        y2 = np.array(y2)

    idx_0 = np.where(y1 == 0)[0]
    idx_1 = np.where(y1 == 1)[0]

    p0 = metrics.accuracy_score(y1[idx_0], y2[idx_0])
    p1 = metrics.accuracy_score(y1[idx_1], y2[idx_1])

    # return 0.5*(estimate_std(p0,len(idx_0)) + estimate_std(p1,len(idx_1)))
    return 0.5 * np.sqrt(estimate_std(p0, len(idx_0)) + estimate_std(p1, len(idx_1)))


def get_corrvox_gs(data_ts, head_mask, regions):
    # remove GS
    cf_rm = ConfoundsRm(data_ts[head_mask].mean(0).reshape(-1, 1), data_ts[head_mask].T, intercept=False)
    data_ts[head_mask] = cf_rm.transform(data_ts[head_mask].mean(0).reshape(-1, 1), data_ts[head_mask].T).T
    # extract time series
    ts_regions = ts.get_ts(data_ts, regions)
    ts_allvox = data_ts[head_mask]
    # compute correlations
    return ts.corr(ts_regions, ts_allvox)


def get_corrvox(data_ts, head_mask, regions):
    # extract time series
    ts_regions = ts.get_ts(data_ts, regions)
    ts_allvox = data_ts[head_mask]
    # compute correlations
    return ts.corr(ts_regions, ts_allvox)


def get_corrvox_std(data_ts, head_mask, regions):
    # extract time series std
    ts_regions = ts.get_ts(data_ts, regions, metric='std')
    ts_allvox = data_ts[head_mask]
    # compute correlations
    return ts.corr(ts_regions, ts_allvox)


class ConfoundsRm:
    def __init__(self, confounds, data, intercept=True):
        self.fit(confounds, data, intercept)

    def fit(self, confounds, data, intercept=True):
        self.data_dim = data.shape
        if confounds == []:
            print('No confounds')
            self.nconfounds = 0
        else:
            if len(self.data_dim) == 3:
                self.a1, self.a2, self.a3 = data.shape
                data_ = data.reshape((self.a1, self.a2 * self.a3))
            elif len(self.data_dim) == 4:
                self.a1, self.a2, self.a3, self.a4 = data.shape
                data_ = data.reshape((self.a1, self.a2 * self.a3 * self.a4))
            else:
                data_ = data
            self.nconfounds = confounds.shape[1]
            self.reg = linear_model.LinearRegression(fit_intercept=intercept)
            # print data_.shape,confounds.shape
            self.reg.fit(confounds, data_)

    def transform(self, confounds, data):
        # compute the residual error
        if self.nconfounds == 0:
            return data
        else:
            if len(data.shape) == 3:
                data_ = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
                res = data_ - self.reg.predict(confounds)
                return res.reshape((data.shape[0], data.shape[1], data.shape[2]))
            elif len(data.shape) == 4:
                data_ = data.reshape((data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))
                res = data_ - self.reg.predict(confounds)
                return res.reshape((data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
            else:
                data_ = data
                return data_ - self.reg.predict(confounds)

    def transform_batch(self, confounds, data, batch_size=50):
        # compute the residual error
        if self.nconfounds == 0:
            return data
        else:
            # batch convert the data
            nbatch = data.shape[0] / (batch_size)  # number of batch
            batch_res = []
            for idx_batch in range(nbatch):
                if idx_batch == nbatch - 1:
                    batch_res.append(
                        self.transform(confounds[idx_batch * batch_size:-1, ...], data[idx_batch * batch_size:-1, ...]))
                else:
                    batch_res.append(self.transform(confounds[idx_batch * batch_size:(1 + idx_batch) * batch_size, ...],
                                                    data[idx_batch * batch_size:(1 + idx_batch) * batch_size, ...]))
            return np.vstack(batch_res)

    def nConfounds(self):
        return self.nconfounds

    def intercept(self):
        if len(self.data_dim) == 3:
            return self.reg.intercept_.reshape((1, self.data_dim[1], self.data_dim[2]))
        elif len(self.data_dim) == 4:
            return self.reg.intercept_.reshape((1, self.data_dim[1], self.data_dim[2], self.data_dim[3]))
        else:
            return self.reg.intercept_


def compute_acc_noconf(x, y, verbose=False, balanced=True, loo=False, nfolds=10, gs_kfolds=5, optimize=True, C=.01):
    return compute_acc_conf(x, y, [], verbose, balanced, loo, nfolds, gs_kfolds, optimize, C)


def compute_acc_conf(x, y, confounds, verbose=False, balanced=True, loo=False, nfolds=10, gs_kfolds=5, optimize=True,
                     C=.01):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)

    if loo:
        cv = LeaveOneOut(len(y))
    else:
        cv = StratifiedKFold(y=encoder.transform(y), n_folds=nfolds)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    total_test_score = []
    y_pred = []
    # clf_array = []
    bc_all = []

    prec = []
    recall = []

    if len(np.unique(y)) == 1:
        print('Unique class: 100%', np.sum(encoder.transform(y) == 0) / len(y))
        return (1., 0., len(y))

    for i, (train, test) in enumerate(cv):

        select_x = x.copy()

        # betacluster = bc.BetaCluster(crm.transform(confounds[train,:],select_x[train,:]),encoder.transform(y[train]),100,k_feature=200)
        # bc_all.append(betacluster)

        if balanced:
            clf = SVC(kernel='linear', class_weight='balanced', C=C)
        else:
            clf = SVC(kernel='linear', C=C)

        if len(confounds) == 0:
            xtrain = select_x[train, :]
            xtest = select_x[test, :]
        else:
            crm = ConfoundsRm(confounds[train, :], select_x[train, :])
            xtrain = crm.transform(confounds[train, :], select_x[train, :])
            xtest = crm.transform(confounds[test, :], select_x[test, :])

        ytrain = encoder.transform(y[train])
        ytest = encoder.transform(y[test])

        # clf.probability = True
        if optimize:
            clf, score = plib.grid_search(clf, xtrain, ytrain, n_folds=gs_kfolds, verbose=verbose)

        clf.fit(xtrain, ytrain)
        total_test_score.append(clf.score(xtest, ytest))
        # clf_array.append(clf)

        prec.append(metrics.precision_score(ytest, clf.predict(xtest)))
        recall.append(metrics.recall_score(ytest, clf.predict(xtest)))

        if loo:
            y_pred.append(clf.predict(xtest))
        if verbose:
            print('nSupport: ', clf.n_support_)
            print("Train:", clf.score(xtrain, ytrain))
            print("Test :", clf.score(xtest, ytest))
            print("Prediction :", clf.predict(xtest))
            print("Real Labels:", ytest)
            print('Precision:', prec[-1], 'Recall:', recall[-1])
    y_pred = np.array(y_pred)[:, 0]
    if loo:
        total_std_test_score = estimate_std(metrics.accuracy_score(encoder.transform(y), np.array(y_pred)), len(y))
        print(
        'Mean:', np.mean(total_test_score), 'Std:', total_std_test_score, 'AvgPrecision:', np.mean(prec), 'AvgRecall:',
        np.mean(recall))
        return [np.mean(total_test_score), total_std_test_score, len(y), y_pred]
    else:
        print('Mean:', np.mean(total_test_score), 'Std:', np.std(total_test_score), 'AvgPrecision:', np.mean(prec),
              'AvgRecall:', np.mean(recall))
        return [np.mean(total_test_score), np.std(total_test_score), len(y)]


def sv_metric(n, nsv):
    return nsv / float(n)
    # return (n-nsv)/float(n) #lower the n sv is greater the score


def get_opt_model(x, y):
    # grid search and SVM
    clf = svm.SVC(kernel='rbf', class_weight='balanced')
    clf.probability = True
    # clf = svm.SVC(kernel='rbf')
    clf, best_score = plib.grid_search(clf, x, y, n_folds=10, verbose=False)
    clf.fit(x, y)
    return clf


def basicconn(skf, X, y):
    total_score = 0
    for train_index, test_index in skf:
        # print("TRAIN:", train_index, "TEST:", test_index)
        # Feature selection
        # selectf = SelectFpr().fit(X[train_index],y[train_index])
        # selectf = SelectKBest(f_classif, k=750).fit(X[train_index],y[train_index])
        # tmp_x = selectf.transform(X[train_index])
        # Train
        # clf = RandomForestClassifier(n_estimators=20)
        # clf = clf.fit(tmp_x, y[train_index])
        # clf.feature_importances_
        # SVM
        # clf = svm.LinearSVC()
        # clf = svm.SVC()
        # clf.fit(tmp_x, y[train_index])
        clf = plib.classif(X[train_index], y[train_index])
        # clf.support_vec()
        # Test
        # pred = clf.predict(selectf.transform(X[test_index]))
        pred = clf.predict(X[test_index])
        print("Target     : ", y[test_index])
        print("Prediction : ", pred)
        matchs = np.equal(pred, y[test_index])
        score = np.divide(np.sum(matchs), np.float64(matchs.size))
        total_score = score + total_score
    return np.divide(total_score, skf.n_folds)


def splitconn(skf, X, y):
    total_score = 0
    for train_index, test_index in skf:
        # Train
        clf1 = plib.classif(X[train_index, 0:2475:1], y[train_index])
        clf2 = plib.classif(X[train_index, 2475:4950:1], y[train_index])
        pred1 = clf1.decision_function(X[train_index, 0:2475:1])
        pred2 = clf2.decision_function(X[train_index, 2475:4950:1])
        clf3 = svm.SVC()
        y[train_index].shape
        np.array([pred1, pred2])
        clf3.fit(np.array([pred1, pred2]).transpose(), y[train_index])
        # clf3 = plib.classif(np.matrix([pred1,pred2]).transpose(),y[train_index])

        # Test
        pred1 = clf1.decision_function(X[test_index, 0:2475:1])
        pred2 = clf2.decision_function(X[test_index, 2475:4950:1])
        predfinal = clf3.predict(np.matrix([pred1, pred2]).transpose())
        print("Target     : ", y[test_index])
        print("Prediction : ", predfinal)
        matchs = np.equal(predfinal, y[test_index])
        score = np.divide(np.sum(matchs), np.float64(matchs.size))
        total_score = score + total_score
    return np.divide(total_score, skf.n_folds)


def multisplit(skf, X, y, stepsize=1000):
    total_score = 0
    for train_index, test_index in skf:
        wl = []
        pred1 = np.matrix([])
        # Training
        for x in range(0, len(X[0]), stepsize):
            clf1 = plib.classif(X[train_index, x:x + stepsize], y[train_index])
            tmp_p = np.matrix(clf1.decision_function(X[train_index, x:x + stepsize]))
            if pred1.size == 0:
                pred1 = tmp_p
            else:
                pred1 = np.concatenate((pred1, tmp_p), axis=1)
            wl.append(clf1)
        # selectf = SelectKBest(f_classif, k=5).fit(pred1, y[train_index])
        selectf = SelectFpr().fit(pred1, y[train_index])
        clf3 = AdaBoostClassifier(n_estimators=100)
        # clf3 = svm.SVC(class_weight='balanced')
        # clf3 = RandomForestClassifier(n_estimators=20)
        clf3.fit(selectf.transform(pred1), y[train_index])
        # Testing
        predtest = np.matrix([])
        k = 0
        for x in range(0, len(X[0]), stepsize):
            tmp_p = np.matrix(wl[k].decision_function(X[test_index, x:x + stepsize]))
            if predtest.size == 0:
                predtest = tmp_p
            else:
                predtest = np.concatenate((predtest, tmp_p), axis=1)
            k += 1
        # Final prediction
        predfinal = clf3.predict(selectf.transform(predtest))
        print("Target     : ", y[test_index])
        print("Prediction : ", predfinal)
        matchs = np.equal(predfinal, y[test_index])
        score = np.divide(np.sum(matchs), np.float64(matchs.size))
        total_score = score + total_score
    return np.divide(total_score, skf.n_folds)
