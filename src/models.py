# -*- coding: utf-8 -*-
import sys
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

reload(sys)
sys.setdefaultencoding('utf-8')

class Models():
    lrclf = LogisticRegression()
    svmclf = svm.SVC(kernel='linear')
    rfclf = RandomForestClassifier()
    
    def __init__(self):
        pass
    
    def lrdemo(self, X, y):
        self.lrclf.fit(X, y)
        return self.lrclf
        
    def svmdemo(self, X, y):
        self.svmclf.fit(X, y)
        return self.svmclf
        
    def rfdemo(self, X, y):
        self.rfclf.fit(X, y)
        return self.rfclf