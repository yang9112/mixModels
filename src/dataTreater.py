# -*- coding: utf-8 -*-
import re
import csv
import xlrd

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
            if row_values[0] == '' or row_values[4] == '':
                continue
            
            title_data.append(row_values[1])
            content_data.append(re.sub('<.*?>', '', row_values[2]))
            
            if row_values[3] == '':
                result_data.append(row_values[4])
            else:
                result_data.append(row_values[3])
        
        return title_data, content_data, result_data
    
    def getTestExcel(self, filename):
        reader = xlrd.open_workbook(filename)
        
        table = reader.sheets()[1]
        title_data = []
        content_data = []
        
        for i in range(table.nrows):
            row_values = table.row_values(i)
            if row_values[4] != '':
                continue
            
            title_data.append(row_values[1])
            content_data.append(re.sub('<.*?>', '', row_values[2]))
            
        return title_data, content_data
            
            
if __name__ == '__main__':
    DT = DataTreater()
    [title_data, content_data, result_data] = DT.readExcel('train.xlsx')