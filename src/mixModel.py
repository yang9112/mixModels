# -*- coding: utf-8 -*-
import sys
import time
import random
import numpy as np

from dataTreater import DataTreater
from preTreater import PreTreater
from models import Models
from sklearn import cross_validation
from sklearn.externals import joblib

reload(sys)
sys.setdefaultencoding('utf-8')

class MixModel():
    DT = DataTreater()

    def __init__(self,
                 dataFile = '../data/data.npy',
                 dataFileTrain = '../data/trainData.npy',
                 dataFileTest = '../data/testData.npy',
                 dataDictFile = '../data/dataDict.npy',
                 originDataFile = '../data/data.xlsx',
                 lrModelFile = '../models/lrModel',
                 svmModelFile = '../models/svmModel',
                 rfModelFile = '../models/rfModel'
                 ):
                     
        self.dataFile = dataFile
        self.dataFileTrain = dataFileTrain
        self.dataFileTest = dataFileTest
        self.dataDictFile = dataDictFile
        self.originDataFile = originDataFile
        self.lrModelFile = lrModelFile
        self.svmModelFile = svmModelFile
        self.rfModelFile = rfModelFile
    
    def pretreatment(self):
        #read data
        [title, content, result] = self.DT.readExcel(self.originDataFile)

        for i in range(len(result)):
            if result[i] < 0:
                result[i] = -1
            
        PT = PreTreater()
        keydata = PT.getKeywords(content)
        datadict = PT.getDict()
        trainData = PT.createTrainData_withdict(datadict, keydata)

        np.save(self.dataFile, [trainData, result])
        np.save(self.dataDictFile, [datadict])
        self.split_train_test([trainData, result])

    def split_train_test(self, data = '', rs=0):
        if data:
            [trainData, result] = data
        else:
            [trainData, result] = np.load(self.dataFile)

        [x_tr, x_te, y_tr, y_te] = cross_validation.train_test_split(trainData,
                            result, test_size = 0.2, random_state=rs)

        np.save(self.dataFileTrain, [x_tr, y_tr])
        np.save(self.dataFileTest, [x_te, y_te])
        
    def build_model(self, save_log = True):
        #load data
        [x_tr, y_tr] = np.load(self.dataFileTrain)
        
        #train model
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
        [datadict] = np.load(self.dataDictFile)
        
        try:       
            lrclf = joblib.load(self.lrModelFile)
            svmclf = joblib.load(self.svmModelFile)
            rfclf = joblib.load(self.rfModelFile)
        except:
            print 'load model failed! please ensure the model is exist.'
            sys.exit(1)
        
        self.cross_test(lrclf, svmclf, rfclf)
        #self.test(datadict, lrclf, svmclf, rfclf)
            
    def demo(self):
        pass

    def test(self, datadict, lrclf, svmclf, rfclf):
        [title, content, result] = self.DT.readExcel(self.originDataFile)
        PT = PreTreater()
        keydata = PT.getKeywords(content[0:150])
        testData = PT.createTrainData_withdict(datadict, keydata)
        
        fp = open('for_test.log', 'wb')        
        
        lr_result = lrclf.predict(testData)
        svm_result = svmclf.predict(testData)
                
        rf_result = rfclf.predict(np.column_stack((lr_result, svm_result)))

        score = 0.0       
        for i in range(150):
            if list(rf_result)[i] - result[i] == 0:
                score = score + 1
        print score/150
        
        for i in range(150):       
            fp.write(title[i].encode('utf8') + ' ' + str(rf_result[i]) + '\n')
            
        fp.close()
        
    def cross_test(self, lrclf, svmclf, rfclf):
        [testData, result] = np.load(self.dataFileTest)
        lr_result = lrclf.predict(testData)
        svm_result = svmclf.predict(testData)
        
        rf_result = rfclf.predict(np.column_stack((lr_result, svm_result)))
        
        precision = 0.0
        precision_abs = 0.0
        
        for i in range(rf_result.shape[0]):
            gap = list(rf_result)[i] - result[i]
            if gap == 0:
                precision = precision + 1
            if abs(gap) <= 1:
                precision_abs = precision_abs + 1
        print precision/rf_result.shape[0]
        print precision_abs/rf_result.shape[0]
        #print rfclf.score(np.column_stack((lr_result, svm_result)), result)
        
if __name__ == '__main__':
#    dataFile = '../data/data.npy'
#    dataFileTrain = '../data/trainData.npy'
#    dataFileTest = '../data/testData.npy'
#    dataDictFile = '../data/dataDict.npy'
#    originDataFile = '../data/data.xlsx'
#    lrModelFile = '../models/lrModel'
#    svmModelFile = '../models/svmModel'
#    rfModelFile = '../models/rfModel'
    
    MM = MixModel()
    start = time.time()
    #MM.pretreatment()
#    print 'Pretreatment Cost: %s second' % str(time.time() - start)
    for i in range(5):
        MM.split_train_test(rs=random.randint(1,99))
        MM.build_model()
        print 'build model Cost: %s second' % str(time.time() - start)
        MM.predict()
        print 'predict model Cost: %s second' % str(time.time() - start)