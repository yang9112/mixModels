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
    """
    modelType:
    0: lrModel, svmModel
    1: lrModel, svmModel, dictModel
    2: lrModel, lrTModel, svmModel
    3: lrModel, lrTModel, svmModel, dictModel
    ...
    """
    modelType = 0

    def __init__(self,
                 modelType = 0,
                 dataFile = '../data/data.npy',
                 dataFileTrain = '../data/trainData.npy',
                 dataFileTitleTrain = '../data/trainTitleData.npy',
                 dataFileTest = '../data/testData.npy',
                 dataFileTitleTest = '../data/testTitleData.npy',
                 dataIndex = '../data/dataIndex.npy',
                 dataDictFile = '../data/dataDict.npy',
                 originDataFile = '../data/data.xlsx',
                 dictResult = '../data/dictResult.npy',
                 lrModelFile = '../models/lrModel',
                 lrTModelFile = '../models/lrTModel',
                 svmModelFile = '../models/svmModel',
                 rfModelFile = '../models/rfModel'
                 ):

        self.modelType = modelType
        self.dataFile = dataFile
        self.dataFileTrain = dataFileTrain
        self.dataFileTitleTrain = dataFileTitleTrain
        self.dataFileTest = dataFileTest
        self.dataFileTitleTest = dataFileTitleTest
        self.dataIndex = dataIndex
        self.dataDictFile = dataDictFile
        self.originDataFile = originDataFile
        self.dictResult = dictResult
        self.lrModelFile = lrModelFile
        self.lrTModelFile = lrTModelFile
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

        if self.modelType == 1 or self.modelType == 3:
            keydata_title = PT.getKeywords(title, all_tag=True)
            trainTitleData = PT.createTrainData_withdict(datadict, keydata_title)
            np.save(self.dataDictFile, [datadict])

        np.save(self.dataFile, [trainData, trainTitleData, np.array(result)])
        self.createRandomSeed(len(result))

    def createRandomSeed(self, n_sample):
#        cv = cross_validation.ShuffleSplit(n_sample, n_iter=5,
#                                           test_size = 0.2, random_state=random.randint(0,100))
        sample_list = np.array(range(n_sample))
        random.shuffle(sample_list)
        cv = cross_validation.KFold(n_sample, n_folds=5)
        tt_idx = [[sample_list[m], sample_list[n]] for m,n in cv]

        np.save(self.dataIndex, [tt_idx])

    def createTrainTest(self, idx_id = -1):
        [trainData, trainTitleData, result] = np.load(self.dataFile)
        [tt_idx] = np.load(self.dataIndex)

        if idx_id >= 0 and idx_id < len(tt_idx):
            x_tr = trainData[tt_idx[idx_id][0], :]
            x_tr_title = trainTitleData[tt_idx[idx_id][0], :]
            x_te = trainData[tt_idx[idx_id][1], :]
            x_te_title = trainTitleData[tt_idx[idx_id][1], :]
            y_tr = result[tt_idx[idx_id][0]]
            y_te = result[tt_idx[idx_id][1]]

            np.save(self.dataFileTitleTrain, [x_tr_title, y_tr])
            np.save(self.dataFileTitleTest, [x_te_title, y_te])
            if self.modelType == 1 or self.modelType == 3:
                np.save(self.dataFileTrain, [tt_idx[idx_id][0], x_tr, y_tr])
                np.save(self.dataFileTest, [tt_idx[idx_id][1], x_te, y_te])
            else:
                np.save(self.dataFileTrain, [x_tr, y_tr])
                np.save(self.dataFileTest, [x_te, y_te])
        else:
            x_tr = trainData
            x_te = result
            np.save(self.dataFileTrain, [x_tr, y_tr])

    def build_model(self, save_tag = True):
        #load data
        tr_data = np.load(self.dataFileTrain)
        if self.modelType == 0 or self.modelType == 2:
            [x_tr, y_tr] = tr_data
        elif self.modelType == 1 or self.modelType == 3:
            [tt_idx, x_tr, y_tr] = tr_data
        else:
            print 'modelType Error'
            sys.exit(2)
        [x_tr_title, y_tr] = np.load(self.dataFileTitleTrain)

        #train model
        models = Models()
        lrclf = models.lrdemo(x_tr, y_tr)
        lrclf_title = models.lrdemo(x_tr_title, y_tr)
        svmclf = models.svmdemo(x_tr, y_tr)

        #get the features
        if self.modelType == 0:
            x_tr_second = np.column_stack((lrclf.predict(x_tr), svmclf.predict(x_tr)))
        elif self.modelType == 1:
            dicResult = np.array(np.load(self.dictResult)[0])[tt_idx]
            x_tr_second = np.column_stack((lrclf.predict(x_tr), svmclf.predict(x_tr), dicResult))
        elif self.modelType == 2:
            x_tr_second = np.column_stack((lrclf.predict(x_tr), lrclf_title.predict(x_tr_title),
                                           svmclf.predict(x_tr)))
        else:
            dicResult = np.array(np.load(self.dictResult)[0])[tt_idx]
            x_tr_second = np.column_stack((lrclf.predict(x_tr), lrclf_title.predict(x_tr_title),
                                           svmclf.predict(x_tr), dicResult))

        rfclf = models.rfdemo(x_tr_second, y_tr)

        if save_tag:
            joblib.dump(lrclf, self.lrModelFile)
            joblib.dump(lrclf_title, self.lrTModelFile)
            joblib.dump(svmclf, self.svmModelFile)
            joblib.dump(rfclf, self.rfModelFile)

    def predict(self):
         #load data
        [datadict] = np.load(self.dataDictFile)

        try:
            lrclf = joblib.load(self.lrModelFile)
            lrtclf = joblib.load(self.lrTModelFile)
            svmclf = joblib.load(self.svmModelFile)
            rfclf = joblib.load(self.rfModelFile)
        except:
            print 'load model failed! please ensure the model is exist.'
            sys.exit(1)

        self.cross_test(lrclf, lrtclf, svmclf, rfclf)
        #self.test(datadict, lrclf, svmclf, rfclf)

    def demo(self):
        pass

#    def test(self, datadict, lrclf, svmclf, rfclf):
#        [title, content, result] = self.DT.readExcel(self.originDataFile)
#        PT = PreTreater()
#        keydata = PT.getKeywords(content[0:150])
#        testData = PT.createTrainData_withdict(datadict, keydata)
#
#        fp = open('for_test.log', 'wb')
#
#        lr_result = lrclf.predict(testData)
#        svm_result = svmclf.predict(testData)
#
#        rf_result = rfclf.predict(np.column_stack((lr_result, svm_result)))
#
#        score = 0.0
#        for i in range(150):
#            if list(rf_result)[i] - result[i] == 0:
#                score = score + 1
#        print score/150
#
#        for i in range(150):
#            fp.write(title[i].encode('utf8') + ' ' + str(rf_result[i]) + '\n')
#
#        fp.close()

    def cross_test(self, lrclf, lrtclf, svmclf, rfclf):
        #load data
        tr_data = np.load(self.dataFileTest)
        if self.modelType == 0 or self.modelType == 2:
            [x_te, y_te] = tr_data
        elif self.modelType == 1 or self.modelType == 3:
            [tt_idx, x_te, y_te] = np.load(self.dataFileTest)
        else:
            print 'modelType Error'
            sys.exit(2)

        [x_te_title, y_te] = np.load(self.dataFileTitleTest)

        lr_result = lrclf.predict(x_te)
        lrt_result = lrtclf.predict(x_te_title)
        svm_result = svmclf.predict(x_te)

        if self.modelType == 0:
            rf_result = rfclf.predict(np.column_stack((lr_result, svm_result)))
        elif self.modelType == 1:
            dic_result = np.array(np.load(self.dictResult)[0])[tt_idx]
            rf_result = rfclf.predict(np.column_stack((lr_result, svm_result, dic_result)))
        elif self.modelType == 2:
            rf_result = rfclf.predict(np.column_stack((lr_result, lrt_result, svm_result)))
        elif self.modelType == 3:
            dic_result = np.array(np.load(self.dictResult)[0])[tt_idx]
            rf_result = rfclf.predict(np.column_stack((lr_result, lrt_result, svm_result, dic_result)))

        #evaluate the result of models
        precision = 0.0
        precision_abs = 0.0
        [sum_1, sum_2, sum_3, sum_4, sum_5, sum_6] = np.zeros(6)

        for i in range(rf_result.shape[0]):
            gap = rf_result[i] - y_te[i]
            if gap == 0:
                precision = precision + 1
            if abs(gap) <= 1:
                precision_abs = precision_abs + 1

            if rf_result[i] == -1:
                sum_1 += 1
            if y_te[i] == -1:
                sum_2 += 1
            if rf_result[i] == -1 and y_te[i] == -1:
                sum_3 += 1

            if rf_result[i] == 1:
                sum_4 += 1
            if y_te[i] == 1:
                sum_5 += 1
            if rf_result[i] == 1 and y_te[i] == 1:
                sum_6 += 1

        print '准确率：' + str(precision/rf_result.shape[0])
        print '误差在1之内的准确率：' + str(precision_abs/rf_result.shape[0])

        pre = sum_3/sum_1
        recall = sum_3/sum_2
        F_score =  pre*recall*2/(pre + recall)
        print '负例以及正例个数', sum_3, sum_6
        print '负面消息的准确率：', pre
        print '负面消息的召回率：', recall
        print '负面消息的F值：', F_score

        pre = sum_6/sum_4
        recall = sum_6/sum_5
        F_score =  pre*recall*2/(pre + recall)
        print '正面消息的准确率：' + str(pre)
        print '正面消息的召回率：' + str(recall)
        print '正面消息的F值：' + str(F_score)
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

    MM = MixModel(1)
    start = time.time()
    #MM.pretreatment()
    print 'Pretreatment Cost: %s second' % str(time.time() - start)
    MM.createRandomSeed(922)
    for i in range(5):
        MM.createTrainTest(idx_id = i)
        MM.build_model()
        print 'build model Cost: %s second' % str(time.time() - start)
        MM.predict()
        print 'predict model Cost: %s second' % str(time.time() - start)