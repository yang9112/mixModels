# -*- coding: utf-8 -*-
import sys
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from dictScoreModel import DictScoreModel
from sklearn import cross_validation

reload(sys)
sys.setdefaultencoding('utf-8')

class Models():
    lrclf = LogisticRegression()
    svmclf = svm.SVC(kernel='linear', C=1)
    svmclf_linear = svm.SVC(kernel='linear', C=1)
    rfclf = RandomForestClassifier(n_estimators=200, min_samples_split=1)
    dsmclf = DictScoreModel()
    
    def __init__(self):
        pass

    def select_demo(self, key, X, y):
        if key == 'lrModel' or key == 'lrTModel':
            clf = self.lrdemo(X, y)
        elif key == 'svmModel':
            clf = self.svmdemo(X, y)
        elif key == 'rfModel':
            clf = self.rfdemo(X, y)
        elif key == 'scoreModel':
            clf = self.dsmclf
        else:
            print 'Error: unkown name of key!'
            sys.exit(2)

        return clf

    def lrdemo(self, X, y):
#        scores = cross_validation.cross_val_score(
#            self.lrclf, X, y, cv=5)
#
#        print 'lr_model: ', scores, scores.mean()
        self.lrclf.fit(X, y)
        return self.lrclf

    def svmdemo(self, X, y):
#        scores = cross_validation.cross_val_score(
#            self.svmclf, X, y, cv=5)
#
#        print 'svm_model: ', scores, scores.mean()
        self.svmclf.fit(X, y)
        return self.svmclf

    def rfdemo(self, X, y):
        self.rfclf.fit(X, y)
        return self.rfclf
