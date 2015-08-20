# -*- coding: utf-8 -*-
import re
import csv
import xlrd
import xlwt
import numpy as np

class DataTreater():
    def __init__(self):
        pass
    
    def readCSV(self, filename):
        csvfile = file(filename, 'rb')
        reader = csv.DictReader(csvfile)
        
        data = []
        for line in reader:
            line['content'] = re.sub('<.*?>', '', line['content'])
            data.append(line)
        
        csvfile.close()
        return data
        
    def readExcel(self, filename):
        reader = xlrd.open_workbook(filename)
        
        #仅包括新闻
        table = reader.sheets()[1]
        
        title_data = []
        content_data = []
        result_data = []        
        
        for i in range(table.nrows):
            row_values = table.row_values(i)
            if row_values[0] == 'url':
                continue
            
            title_data.append(row_values[1])
            content_data.append(re.sub('<.*?>', '', row_values[2]))
            
            if row_values[3] != '':
                result_data.append(row_values[3])
        
        return title_data, content_data, result_data
        
    def readDictExcel(self, filename, originfile):
        reader = xlrd.open_workbook(filename)
        reader_origin = xlrd.open_workbook(originfile)
        
        table = reader.sheets()[1]
        table_origin = reader_origin.sheets()[1]
        
        col_index = table.col_values(0)[1:]
        result_data = table.col_values(6)[1:]
        col_index_origin = table_origin.col_values(0)[1:]
        
        dict_result = []        
        for url in col_index_origin:
            dict_result.append(result_data[col_index.index(url)])        

        np.save('../data/dictResult.npy', [dict_result])
#        f = xlwt.Workbook()
#        sheet = f.add_sheet(u'sheet1',cell_overwrite_ok=True)
#        for i in range(len(dict_result)):
#            sheet.write(i, 0, col_index.index(col_index_origin[i]))
#            sheet.write(i, 2, col_index_origin[i])
#            sheet.write(i, 3, col_index[col_index.index(col_index_origin[i])])
#            sheet.write(i, 1, dict_result[i])
#        f.save('../data/excelFile.xls')
        
            
if __name__ == '__main__':
    DT = DataTreater()
    [title_data, content_data, result_data] = DT.readExcel('../data/data.xlsx')
    DT.readDictExcel('../data/dicModelResult.xlsx', '../data/data.xlsx')