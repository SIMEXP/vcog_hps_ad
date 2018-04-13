#!/usr/bin/python
import numpy as np
from sklearn.model_selection import StratifiedKFold,LeaveOneOut
from sklearn import preprocessing

from sklearn import svm
from sklearn.model_selection import GridSearchCV

from collections import Counter
from sklearn.metrics import accuracy_score
class bagging:
    'Bagging function to do model averaging from various trainned models'
    def __init__(self):
        # do notting
        self.models = []

    def add(self,clf):
        self.models.append(clf)

    def predict(self,x_all):
        bag_vote = []
        pred_matrix = []
        for i, clf in enumerate(self.models):
            if i == 0:
                pred_matrix = clf.predict(x_all[i])
            else:
                pred_matrix = np.vstack((pred_matrix,clf.predict(x_all[i])))

        for i in range(0,pred_matrix.shape[1]):
            c = Counter(pred_matrix[:,i])
            bag_vote.append(c.most_common(1)[0][0])
        return bag_vote
        
    def score(self,x_all,y):
        return accuracy_score(y,self.predict(x_all))

    def predict_proba(self,x_all):
        'Compute the average of the predicted proba of each classifier'
        pred_matrix = []
        for i, clf in enumerate(self.models):
            if i == 0:
                pred_matrix = clf.predict_proba(x_all[i])
            else:
                pred_matrix = pred_matrix + clf.predict_proba(x_all[i])
        
        pred_matrix /= float(len(self.models))
        return pred_matrix


def grid_search(clf, x, y, n_folds=10, verbose=True, detailed=False):
        """
        # Train classifier
        #
        # For an initial search, a logarithmic grid with basis
        # 10 is often helpful. Using a basis of 2, a finer
        # tuning can be achieved but at a much higher cost.
        """
        if verbose:
            print("Running grid search ...")

        #C_range = (10.0 ** np.arange(-2, 3))
        if detailed:
            C_range = np.arange(0.0005, 0.02,0.001)
        else:
            C_range = (10.0 ** np.arange(0.5,-2,-0.25))
        gamma_range = (0)
        if hasattr(clf,'kernel'):
            if clf.kernel != 'linear':
                gamma_range = (10.0 ** np.arange(-3, 1, 0.05)) 
                param_grid = dict(gamma=gamma_range, C=C_range)
            else:
                param_grid = dict(C=C_range)

            if n_folds==1:
                cv = LeaveOneOut()
            else:
                cv = StratifiedKFold(n_splits=n_folds)
            #cv = cross_validation.LeaveOneOut(len(y))
            #grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, n_jobs=-1, scoring='f1')
            grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, n_jobs=-1)
            grid.fit(x, y)
            if verbose:
                print("The best classifier is: ", grid.best_estimator_)
        
            return grid.best_estimator_, grid.best_score_
        else:
            print('No kernel to optimize!')
            return clf

class classif:
    'Prediction tool for multiscale functional neuro imaging'
    empCount = 0

    def __init__(self, x, y, n=10):
        # Feature selection
        #self.selectf = SelectFpr().fit(x,y)
        #if len(x[0]) <= n: n=len(x[0])
        #self.selectf = SelectKBest(f_classif, k=n).fit(x, y)
        #self.tmp_x = self.selectf.transform(x)
        
        # Normalization
        self.scaler = preprocessing.StandardScaler().fit(x)
      
        # Train
        #self.clf = RandomForestClassifier(n_estimators=50)
        #clf = clf.fit(tmp_x, y[train_index])
        #clf.feature_importances_
        # SVM
        #self.clf = LDA()
        #self.clf = svm.LinearSVC(class_weight='balanced')
        self.clf = svm.SVC(kernel='rbf', class_weight='balanced')
        self.clf = grid_search(self.clf, self.scaler.transform(x),y)
        self.clf.fit(self.scaler.transform(x), y)
    
    def grid_search(self, clf, x, y, n_folds=10):
        """
        # Train classifier
        #
        # For an initial search, a logarithmic grid with basis
        # 10 is often helpful. Using a basis of 2, a finer
        # tuning can be achieved but at a much higher cost.
        """
        print("Running grid search ...")
        
        C_range = (10.0 ** np.arange(-5, 3))
        gamma_range = (10.0 ** np.arange(-5, 3))
        param_grid = dict(gamma=gamma_range, C=C_range)
        
        cv = StratifiedKFold(y=y, n_folds=n_folds)
        grid = GridSearchCV(clf, param_grid=param_grid, cv=cv)
        grid.fit(x, y)
        print("The best classifier is: ", grid.best_estimator_)
        return grid.best_estimator_

    def predict(self, x):
        # Test
        #x_select = self.selectf.transform(x)
        x_select = self.scaler.transform(x)
        if len(x_select[0]) != 0:
            pred = self.clf.predict(x_select)
            #print "Prediction : ", pred
            return pred

    def decision_function(self, x):
        # return the decision function
        x_select = self.selectf.transform(x) 
        x_select = self.scaler.transform(x_select)
        if len(x_select[0]) != 0:
            df = self.clf.decision_function(x_select)
            return df
        else:
            return []
            print("ZERO!!")
        #print "Decision function : ", df
        #return df

    def support_vec(self):
        # get indices of support vectors
        idx_svec = self.clf.support_
        idx_global = self.selectf.get_support(True)
        return idx_global[idx_svec]

