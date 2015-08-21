# -*- coding: utf-8 -*-
import sys
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation


reload(sys)
sys.setdefaultencoding('utf-8')

class Models():
    lrclf = LogisticRegression()
    svmclf = svm.SVC(kernel='rbf', C=1.5)
    rfclf = RandomForestClassifier(n_estimators=200, min_samples_split=1, max_depth=2)

    def __init__(self):
        pass

    def lrdemo(self, X, y):
#        scores = cross_validation.cross_val_score(
#            self.lrclf, X, y, cv=5)
#
#        print 'lr_model: ', scores, scores.mean()
        self.lrclf.fit(X, y)
        return self.lrclf

    def svmdemo(self, X, y):
        scores = cross_validation.cross_val_score(
            self.svmclf, X, y, cv=5)

        print 'svm_model: ', scores, scores.mean()
        self.svmclf.fit(X, y)
        return self.svmclf

    def rfdemo(self, X, y):
        self.rfclf.fit(X, y)
        return self.rfclf