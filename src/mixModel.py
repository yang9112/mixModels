# -*- coding: utf-8 -*-
import sys
import numpy as np
import jieba.posseg as pseg

from dataTreater import DataTreater
from preTreater import PreTreater
from models import Models
from sklearn import cross_validation
from scipy.sparse import csr_matrix
from sklearn.externals import joblib

reload(sys)
sys.setdefaultencoding('utf-8')

class MixModel():
    DT = DataTreater()

    def __init__(self, dataFile = '../data/trainData.npy', 
                 dataDictFile = '../data/trainDict.npy',
                 originDataFile = '../data/train.xlsx',
                 lrModelFile = '../models/lrModel',
                 svmModelFile = '../models/svmModel',
                 rfModelFile = '../models/rfModel'
                 ):
                     
        self.dataFile = dataFile
        self.dataDictFile = dataDictFile
        self.originDataFile = originDataFile
        self.lrModelFile = lrModelFile
        self.svmModelFile = svmModelFile
        self.rfModelFile = rfModelFile
    
    def pretreatment(self):
        #read data
        [title, content, result] = self.DT.readExcel(self.originDataFile)
        PT = PreTreater()
        keydata = PT.getKeywords(content)
        traindict = PT.getDict()

        trainData = PT.createTrainData_withdict(traindict, keydata)
        
        np.save(self.dataFile, [trainData, result])
        np.save(self.dataDictFile, [traindict])
        
    def build_model(self, save_log = True):
        #load data
        [trainData, result] = np.load(self.dataFile)
        
        for i in range(len(result)):
            if result[i] < 0:
                result[i] = -1
                
        #train model
        [x_tr, x_te, y_tr, y_te] = cross_validation.train_test_split(trainData,
                                    result, test_size = 0.1, random_state=2)
        models = Models()                                    
        lrclf = models.lrdemo(x_tr, y_tr)
        svmclf = models.svmdemo(x_tr, y_tr)
        
        x_tr_second = []
        for i in range(x_tr.shape[0]):        
            x_tr_second.append(list(lrclf.predict(x_tr[i])) + list(svmclf.predict(x_tr[i])))
        #x_tr_second = np.array(x_tr_second)
        rfclf = models.rfdemo(x_tr_second, y_tr)
        
        if save_log:
            joblib.dump(lrclf, self.lrModelFile)
            joblib.dump(svmclf, self.svmModelFile)
            joblib.dump(rfclf, self.rfModelFile)        
        
    def predict(self):
         #load data
        [traindict] = np.load(self.dataDictFile)
        
        try:       
            lrclf = joblib.load(self.lrModelFile)
            svmclf = joblib.load(self.svmModelFile)
            rfclf = joblib.load(self.rfModelFile)
        except:
            print 'load model failed! please ensure the model is exist.'
            sys.exit(1)
        
        
        self.test(traindict, lrclf, svmclf, rfclf)
        
    def demo(self):
        #load data
        [trainData, result] = np.load(self.dataFile)
        [traindict] = np.load(self.dataDictFile)
        
        for i in range(len(result)):
            if result[i] < 0:
                result[i] = -1
                
        #train model
        [x_tr, x_te, y_tr, y_te] = cross_validation.train_test_split(trainData,
                                    result, test_size = 0.2, random_state=2)
        models = Models()                                    
        clf = models.demo(x_tr, y_tr)
        
#        for i in range(len(y_te)):
#            print clf.predict(x_te[i,:]), y_te[i]
            
        self.test(traindict, clf)

    def test(self, traindict, lrclf, svmclf, rfclf):
        [title, content] = self.DT.getTestExcel(self.originDataFile)
        
        for i in range(100):
            seg_list = pseg.cut(content[i].encode('utf8'))
            keys = []
            data = []
            for w in seg_list:
                if w.flag in ['a', 'e', 'u', 'vn', 'vd', 'v', 'i', 'an', 'z']:
                    try:
                        keys.append(traindict[w.word.encode('utf8')])
                        data.append(1)  
                    except:
                        continue
            pred_vector = csr_matrix((data, keys, [0,len(keys)]), shape=(1, len(traindict)), dtype=int)
            print rfclf.predict(list(lrclf.predict(pred_vector)) + 
                                list(svmclf.predict(pred_vector)))

if __name__ == '__main__':
#    dataFile = '../data/trainData.npy'
#    dataDictFile = '../data/trainDict.npy'
#    originDataFile = '../data/train.xlsx'
#    lrModelFile = '../models/lrModel'
#    svmModelFile = '../models/svmModel'
    
    MM = MixModel()
    #MM.pretreatment()
    #MM.demo()
    #MM.build_model()
    MM.predict()