# -*- coding: utf-8 -*-
import sys
import numpy as np
import jieba.posseg as pseg

from dataTreater import DataTreater
from preTreater import PreTreater
from LR_model import LR_Model
from sklearn import cross_validation
from scipy.sparse import csr_matrix


reload(sys)
sys.setdefaultencoding('utf-8')

class MixModel():
    DT = DataTreater()

    def __init__(self):
        pass
    
    def pretreatment(self, filename='trainData.npy'):
        #read data
        [title, content, result] = self.DT.readExcel('train.xlsx')
        PT = PreTreater()
        keydata = PT.getKeywords(content)
        traindict = PT.getDict()

        trainData = PT.createTrainData_withdict(traindict, keydata)
        
        np.save(filename, [trainData, traindict, result])        
        
    def build_model(self):
        pass
    
    def predict(self):
        pass
    
    def demo(self):
        #load data
        [trainData, traindict, result] = np.load('trainData.npy')
        
        for i in range(len(result)):
            if result[i] < 0:
                result[i] = -1
                
        #train model
        [x_tr, x_te, y_tr, y_te] = cross_validation.train_test_split(trainData,
                                    result, test_size = 0.2, random_state=2)
        LR = LR_Model()                                    
        clf = LR.demo(x_tr, y_tr)
        
#        for i in range(len(y_te)):
#            print clf.predict(x_te[i,:]), y_te[i]
            
        self.test(traindict, clf)

    def test(self, traindict, clf):
        [title, content] = self.DT.getTestExcel('train.xlsx')
        
        seg_list = pseg.cut(content[298].encode('utf8'))
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
        print clf.predict(pred_vector)

if __name__ == '__main__':
    MM = MixModel()
    #MM.pretreatment('trainData.npy')
    MM.demo()
    #get train data    