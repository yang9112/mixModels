# -*- coding: utf-8 -*-

import jieba
import jieba.posseg as pseg
from scipy.sparse import csr_matrix

class PreTreater():
    def __init__(self):
        pass

    def getKeywords(self, tables, all_tag = False):
        keys = []
        for content in tables:
            seg_list = pseg.cut(content.encode('utf8'))
            #print '/'.join(seg_list)
            seg = []
            if not all_tag:
                for w in seg_list:
                    if w.flag in ['a', 'e', 'u', 'vn', 'vd', 'v', 'i', 'an', 'z']:
                        seg.append(w.word.encode('utf8'))
            else:
                for w in seg_list:
                    seg.append(w.word.encode('utf8'))

            keys.append(list(set(seg)))
            #keys.append(seg)

        return keys

    def getDict(self):
        directory = dict()
        fp = open(jieba.DEFAULT_DICT, 'rb')
        #fp = open('../data/score.txt', 'rb')
        for line in fp.readlines():
            if line.split()[2] == 'n':
                continue
            directory.setdefault(line.split(' ')[0], len(directory))
        fp.close()

        return directory

    def createTrainDataDict(self, directory, keyData):
        indptr = [0]
        indices = []
        data = []

        for d in keyData:
            for term in d:
                if term not in directory:
                    continue
                index = directory.setdefault(term, len(directory))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))

        return csr_matrix((data, indices, indptr),
                          shape=(len(indptr) - 1, len(directory)), dtype=float)

    def createTrainData(self, keyData):
        indptr = [0]
        indices = []
        data = []
        vocabulary = {}

        for d in keyData:
            for term in d:
                index = vocabulary.setdefault(term, len(vocabulary))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))

        return vocabulary, csr_matrix((data, indices, indptr), dtype=float)

if __name__ == '__main__':
    from dataTreater import DataTreater
    DT = DataTreater()
    [title, content, result] = DT.readExcel('../data/data.xlsx')
    PT = PreTreater()
    keydata = PT.getKeywords(content)
    traindict = PT.getDict()

    #trainData = PT.createTrainDataDict(traindict, keydata)