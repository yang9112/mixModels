# -*- coding: utf-8 -*-
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

reload(sys)
sys.setdefaultencoding('utf-8')

class LR_Model():
    def __init__(self, clf = LogisticRegression()):
        self.clf = clf
    
    def demo(self, X, y):
        self.clf.fit(X, y)
        
        return self.clf
        
    def save_model(self, filename):
        joblib.dump(self.clf, filename)
    
    def load_model(self, filename):
        self.clf = joblib.load(filename)