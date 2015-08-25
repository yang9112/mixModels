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
    svmclf = svm.SVC(kernel='linear', C=1)
    svmclf_linear = svm.SVC(kernel='linear', C=1)
    rfclf = RandomForestClassifier(n_estimators=200, min_samples_split=1)

    def __init__(self):
        pass

    def selectDemo(self, key, X, y):
        if key == 'lrModel' or key == 'lrTModel':
            clf = self.lrDemo(X, y)
        elif key == 'svmModel':
            clf = self.svmDemo(X, y)
        elif key == 'rfModel':
            clf = self.rfDemo(X, y)
        else:
            print 'Error: unkown name of key!'
            sys.exit(2)

        return clf

    def lrDemo(self, X, y):
#        scores = cross_validation.cross_val_score(
#            self.lrclf, X, y, cv=5)
#
#        print 'lr_model: ', scores, scores.mean()
        self.lrclf.fit(X, y)
        return self.lrclf

    def svmDemo(self, X, y):
#        scores = cross_validation.cross_val_score(
#            self.svmclf, X, y, cv=5)
#
#        print 'svm_model: ', scores, scores.mean()
        self.svmclf.fit(X, y)
        return self.svmclf

    def rfDemo(self, X, y):
        self.rfclf.fit(X, y)
        return self.rfclf
