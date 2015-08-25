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
    model_type:
    0: lrModel, svmModel
    1: lrModel, svmModel, dictModel
    2: lrModel, lrTModel, svmModel
    3: lrModel, lrTModel, svmModel, dictModel
    4: lrModel, lrTmodel, dictModel
    ...
    """
    model_type = 0
    model_dict = {'lrModel':1, 'svmModel':1, 'dictModel':0, 'lrTModel':0}

    def __init__(self,
                 model_type = 0,
                 data_file = '../data/data.npy',
                 data_title_file = '../data/dataTitle.npy',
                 data_file_train = '../data/trainData.npy',
                 data_file_title_train = '../data/trainTitleData.npy',
                 data_file_test = '../data/testData.npy',
                 data_file_title_test = '../data/testTitleData.npy',
                 data_index = '../data/dataIndex.npy',
                 data_dict_file = '../data/dataDict.npy',
                 origin_data_file = '../data/data.xlsx',
                 dict_result = '../data/dictResult.npy',
                 rfmodel_file = '../models/rfModel',
                 model_dir = '../models'
                 ):

        self.model_type = model_type
        self.data_file = data_file
        self.data_title_file = data_title_file
        self.data_file_train = data_file_train
        self.data_file_title_train = data_file_title_train
        self.data_file_test = data_file_test
        self.data_file_title_test = data_file_title_test
        self.data_index = data_index
        self.data_dict_file = data_dict_file
        self.origin_data_file = origin_data_file
        self.dict_result = dict_result
        self.rfmodel_file = rfmodel_file
        self.model_dir = model_dir

        #load model type
        self.set_model_dict(self.model_type)

    def set_model_dict(self, model_type):
        #set the dict to zero
        for key in self.model_dict.keys():
            self.model_dict[key] = 0

        if model_type == 0:
            self.model_dict.update({'lrModel':1, 'svmModel':1})
        elif model_type == 1:
            self.model_dict.update({'lrModel':1, 'svmModel':1, 'dictModel':1})
        elif model_type == 2:
            self.model_dict.update({'lrModel':1, 'svmModel':1, 'lrTModel':1})
        elif model_type == 3:
            self.model_dict.update({'lrModel':1, 'svmModel':1,
                                    'lrTModel':1, 'dictModel':1})
        elif model_type == 4:
            self.model_dict.update({'lrModel':1, 'lrTModel':1, 'dictModel':1})

    def pretreatment(self):
        #read data
        [title, content, result] = self.DT.read_excel(self.origin_data_file)

        for i in range(len(result)):
            if result[i] < 0:
                result[i] = -1

        PT = PreTreater()
        keydata = PT.get_keywords(content)
        datadict = PT.getdict()
        traindata = PT.create_train_data_dict(datadict, keydata)
        #traindata = self.normalize_data(trainData)

        #if self.model_dict['lrTmodel']:
        keydata_title = PT.get_keywords(title, all_tag=True)
        train_title_data = PT.create_train_data_dict(datadict, keydata_title)
        np.save(self.data_dict_file, [datadict])
        np.save(self.data_title_file, [train_title_data])

        np.save(self.data_file, [traindata, np.array(result)])

        self.create_random_seed(len(result))

    def normalize_data(self, data):
        row_sums = data.sum(axis=1)[:, 0]
        row_indices, col_indices = data.nonzero()
        data[row_indices, col_indices] /= row_sums[row_indices].T
        return data

    def create_random_seed(self, n_sample):
        cv = cross_validation.ShuffleSplit(n_sample, n_iter=5,
                                           test_size = 0.2,
                                           random_state=random.randint(0,100))
        tt_idx = [[m,n] for m,n in cv]

#        sample_list = np.array(range(n_sample))
#        random.shuffle(sample_list)
#        cv = cross_validation.KFold(n_sample, n_folds=5)
#        tt_idx = [[sample_list[m], sample_list[n]] for m,n in cv]

        np.save(self.data_index, [tt_idx])

    def create_train_test(self, idx_id = -1):
        if self.model_dict['lrTModel']:
            [train_title_data] = np.load(self.data_title_file)

        [traindata, result] = np.load(self.data_file)
        [tt_idx] = np.load(self.data_index)

        if idx_id >= 0 and idx_id < len(tt_idx):
            x_tr = traindata[tt_idx[idx_id][0], :]
            x_te = traindata[tt_idx[idx_id][1], :]
            y_tr = result[tt_idx[idx_id][0]]
            y_te = result[tt_idx[idx_id][1]]

            np.save(self.data_file_train, [tt_idx[idx_id][0], x_tr, y_tr])
            np.save(self.data_file_test, [tt_idx[idx_id][1], x_te, y_te])

            if self.model_dict['lrTModel']:
                x_tr_title = train_title_data[tt_idx[idx_id][0], :]
                x_te_title = train_title_data[tt_idx[idx_id][1], :]
                np.save(self.data_file_title_train, [x_tr_title, y_tr])
                np.save(self.data_file_title_test, [x_te_title, y_te])
        else:
            x_tr = traindata
            x_te = result
            np.save(self.data_file_train, [x_tr, y_tr])

    def build_model(self, save_tag = True):
        #load data
        tr_data = np.load(self.data_file_train)
        [tt_idx, x_tr, y_tr] = tr_data

        if self.model_dict['lrTModel']:
            [x_tr_title, y_tr] = np.load(self.data_file_title_train)
        if self.model_dict['dictModel']:
            dicResult = np.array(np.load(self.dict_result)[0], dtype=float)[tt_idx]

        mid = int(x_tr.shape[0]/2)
        #train model
        models = Models()

        #get the features
        n_features = []
        for key in self.model_dict.keys():
            #remove the models unused
            if not self.model_dict[key]:
                continue

            x_train = x_tr
            if key == 'dictModel':
                n_features.append(dicResult)
                continue
            elif key == 'lrTModel' and self.model_dict[key]:
                x_train = x_tr_title

            if self.model_dict[key]:
                clf = models.select_demo(key, x_train[:mid, :], y_tr[:mid])
                mid_to_end = list(clf.predict(x_train[mid:,:]))
                clf = models.select_demo(key, x_train[mid:, :], y_tr[mid:])
                top_to_mid = list(clf.predict(x_train[:mid, :]))
                n_features.append(np.array(top_to_mid + mid_to_end))
                clf = models.select_demo(key, x_train, y_tr)
                if save_tag:
                    joblib.dump(clf, self.model_dir + '/' + key)

        if len(n_features) > 1:
            x_tr_second = np.column_stack(tuple(n_features))
        else:
            print 'Error: less models'
            sys.exit(1)

        #train the second model
        rfclf = models.rfdemo(x_tr_second, y_tr)
        if save_tag:
            joblib.dump(rfclf, self.rfmodel_file)

    def predict(self):
         #load models
        rfclf = joblib.load(self.rfmodel_file)

        feature_models = dict()
        for key in self.model_dict.keys():
            if key != 'dictModel' and self.model_dict[key]:
                clf = joblib.load(self.model_dir + '/' + key)
                feature_models.setdefault(key, clf)

        self.cross_test(rfclf, feature_models)
        #[datadict] = np.load(self.data_dict_file)
        #self.test(datadict, lrclf, svmclf, rfclf)

    def demo(self):
        pass

#    def test(self, datadict, lrclf, svmclf, rfclf):
#        [title, content, result] = self.DT.read_excel(self.origin_data_file)
#        PT = PreTreater()
#        keydata = PT.get_keywords(content[0:150])
#        testData = PT.create_train_data_dict(datadict, keydata)
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

    def cross_test(self, rfclf, feature_models, evalTag = True):
        #load data
        [tt_idx, x_te, y_te] = np.load(self.data_file_test)
        if self.model_dict['lrTModel']:
            [x_te_title, y_te] = np.load(self.data_file_title_test)
        if self.model_dict['dictModel']:
            dic_result = np.array(np.load(self.dict_result)[0])[tt_idx]

        #get features
        n_features = []

        for key in self.model_dict.keys():
            if self.model_dict[key]:
                x_train = x_te
                if key == 'dictModel':
                    n_features.append(dic_result)
                    continue
                elif key == 'lrTModel':
                    x_train = x_te_title
                n_features.append(feature_models[key].predict(x_train))

        rf_result = rfclf.predict(np.column_stack(tuple(n_features)))

        if evalTag:
            self.evaluate(rf_result, y_te)

        self.DT.write_data('../data/negative.xls', self.origin_data_file,
                          tt_idx[rf_result == -1], y_te[rf_result == -1])
        self.DT.write_data('../data/positive.xls', self.origin_data_file,
                          tt_idx[rf_result == 1], y_te[rf_result == 1])
        self.DT.write_data('../data/zeros.xls', self.origin_data_file,
                          tt_idx[rf_result == 0], y_te[rf_result == 0])

    def evaluate(self, x_te, y_te):
        #evaluate the result of models
        precision = 0.0
        precision_abs = 0.0
        [sum_1, sum_2, sum_3, sum_4, sum_5, sum_6] = np.zeros(6)

        for i in range(x_te.shape[0]):
            gap = x_te[i] - y_te[i]
            if gap == 0:
                precision = precision + 1
            if abs(gap) <= 1:
                precision_abs = precision_abs + 1

            if x_te[i] == -1:
                sum_1 += 1
                if y_te[i] == -1:
                    sum_3 += 1
            elif x_te[i] == 1:
                sum_4 += 1
                if y_te[i] == 1:
                    sum_6 += 1

            if y_te[i] == -1:
                sum_2 += 1
            elif y_te[i] == 1:
                sum_5 += 1

        print '准确率：' + str(precision/x_te.shape[0])
        print '误差在1之内的准确率：' + str(precision_abs/x_te.shape[0])

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
#    data_file = '../data/data.npy'
#    data_file_train = '../data/trainData.npy'
#    data_file_test = '../data/testData.npy'
#    data_dict_file = '../data/dataDict.npy'
#    origin_data_file = '../data/data.xlsx'

    MM = MixModel(4)
    start = time.time()
    #MM.pretreatment()
    print 'Pretreatment Cost: %s second' % str(time.time() - start)
    MM.create_random_seed(922)
    for i in range(5):
        MM.create_train_test(idx_id = i)
        MM.build_model()
        print 'build model Cost: %s second' % str(time.time() - start)
        MM.predict()
        print 'predict model Cost: %s second' % str(time.time() - start)