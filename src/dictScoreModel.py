# -*- coding: utf-8 -*-
import os, sys
from preTreater import PreTreater
import numpy as np

class DictScoreModel():
    wd_score_dict = None
    id_score_dict = None
    wd_score_dict_file = '../data/wd_score_dict.npy'
    id_score_dict_file = '../data/id_score_dict.npy'    
    
    def __init__(self):
        pass
    
    def predict(self, x_te):
        if self.id_score_dict == None:
            if os.path.exists(self.id_score_dict_file):
                [self.id_score_dict] = np.load(self.id_score_dict_file)
            else:
                self.set_id_score_dict(self.id_score_dict_file)
                
        return self.cacul(x_te, self.id_score_dict)
        
    def set_id_score_dict(self, id_score_dict_file):
        if os.path.exists(id_score_dict_file):
            [self.id_score_dict] = np.load(id_score_dict_file)
        else:
            print 'failed to load the file of id_score_dict'
            sys.exit(2)

    def set_wd_score_dict(self, wd_score_dict_file):
        if os.path.exists(wd_score_dict_file):
            [self.wd_score_dict] = np.load(wd_score_dict_file)
        else:
            print 'failed to load the file of wd_score_dict'
            sys.exit(2)
            
    def cacul(self, x_te, id_score_dict):
        sum_score = np.zeros((x_te.shape[0]), dtype=float)
#        sum_score_cp = np.zeros((x_te.shape[0]), dtype=float)
        id_score_vector = np.array(np.array(id_score_dict.values())[:, 1], dtype=float)
        
        #only work for the sparse data
        for row_idx in range(x_te.shape[0]):
            row, col = x_te[row_idx].nonzero()
#            sum_score[row_idx] = np.dot(id_score_vector[col], x_te[row_idx, :].data)
            sum_score[row_idx] = np.sum(id_score_vector[col])

#        pos_mean = np.mean(sum_score[sum_score > 0])
#        neg_mean = np.mean(sum_score[sum_score < 0])
#        sum_score_cp[sum_score > pos_mean] = 1
#        sum_score_cp[sum_score < neg_mean] = -1
        sum_score = sum_score/(max(sum_score) - min(sum_score))
        return sum_score
    
if __name__ == '__main__':
    DSM = DictScoreModel()
    PT = PreTreater()
    wd_score_idx, id_score_idx = PT.get_score_dict('../data/score.txt')
    np.save('../data/id_score_dict.npy', [id_score_idx])    
    from scipy.sparse import csr_matrix
    content = csr_matrix(([1,1,1], [5, 10 ,22], [0,2,3]), shape=((2,100)), dtype=float)
    print DSM.predict(content)
