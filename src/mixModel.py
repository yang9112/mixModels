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
                 testIndex = '../data/testIndex.npy',
                 dataDictFile = '../data/dataDict.npy',
                 originDataFile = '../data/data.xlsx',
                 dictResult = '../data/dictResult.npy',
                 lrModelFile = '../models/lrModel',
                 svmModelFile = '../models/svmModel',
                 rfModelFile = '../models/rfModel'
                 ):
                     
        self.dataFile = dataFile
        self.dataFileTrain = dataFileTrain
        self.dataFileTest = dataFileTest
        self.testIndex = testIndex
        self.dataDictFile = dataDictFile
        self.originDataFile = originDataFile
        self.dictResult = dictResult
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

        cv = cross_validation.ShuffleSplit(len(result), n_iter=5, 
                                           test_size = 0.2, random_state=random.randint(0,100))
        tt_idx = [[m, n] for m,n in cv]
        
        np.save(self.dataFile, [trainData, np.array(result)])
        np.save(self.dataDictFile, [datadict])
        np.save(self.testIndex, [tt_idx])
        
    def createTrainTest(self, idx_id = -1):
        [trainData, result] = np.load(self.dataFile)
        [tt_idx] = np.load(self.testIndex)
        
        if idx_id >= 0 and idx_id < len(tt_idx):            
            x_tr = trainData[tt_idx[idx_id][0], :]
            x_te = trainData[tt_idx[idx_id][1], :]
            y_tr = result[tt_idx[idx_id][0]]
            y_te = result[tt_idx[idx_id][1]]
            np.save(self.dataFileTrain, [tt_idx[idx_id][0], x_tr, y_tr]) 
            np.save(self.dataFileTest, [tt_idx[idx_id][1], x_te, y_te])
        else:
            x_tr = trainData
            x_te = result
            np.save(self.dataFileTrain, [x_tr, y_tr])
        
            
    def build_model(self, save_tag = True):
        #load data
        tr_data = np.load(self.dataFileTrain)
        if len(tr_data) == 2:
            [x_tr, y_tr] = tr_data
        else:
            [tt_idx, x_tr, y_tr] = tr_data
        
        #train model
        models = Models()                                    
        lrclf = models.lrdemo(x_tr, y_tr)
        svmclf = models.svmdemo(x_tr, y_tr)
        
        if len(tr_data) == 2:
            x_tr_second = np.column_stack((lrclf.predict(x_tr), svmclf.predict(x_tr)))
        else:
            dicResult = np.array(np.load(self.dictResult)[0])[tt_idx]
            x_tr_second = np.column_stack((lrclf.predict(x_tr), svmclf.predict(x_tr), dicResult))

        rfclf = models.rfdemo(x_tr_second, y_tr)
        
        if save_tag:
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
        [tt_idx, testData, result] = np.load(self.dataFileTest)
        [dic_result] = np.load(self.dictResult)
        lr_result = lrclf.predict(testData)
        svm_result = svmclf.predict(testData)
        
        dic_result = np.array(dic_result)[tt_idx]
        rf_result = rfclf.predict(np.column_stack((lr_result, svm_result, dic_result)))
        
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
    print 'Pretreatment Cost: %s second' % str(time.time() - start)
    for i in range(5):
        MM.createTrainTest(i)
        MM.build_model()
        print 'build model Cost: %s second' % str(time.time() - start)
        MM.predict()
        print 'predict model Cost: %s second' % str(time.time() - start)