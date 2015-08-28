# -*- coding: utf-8 -*-
import csv
import xlwt
import os, sys
import re
import json
from openpyxl import Workbook

reload(sys)
sys.setdefaultencoding('utf-8')

fp = open('../data/data.jl', 'rb')
#data_file = json.load(fp)

data_dict = []
for line in fp.readlines():
    try:
        data_dict.append(json.loads(line.encode('utf8')))
    except:
        pass
fp.close()

#filename_save = '../data/test_data.csv'
#fp = open(filename_save, 'wb')
#writer = csv.writer(fp)
#
#for item in data_dict:
#    writer.writerow([item['url'], item['title'], re.sub('<.*?>', '', item['content'].encode('utf8'))])
#
#fp.close()

wb = Workbook()
ws = wb.active

for i in range(len(data_dict)):
    item = data_dict[i]
    ws.append([item['url'], item['title'], re.sub('<.*?>', '', item['content'])])
    
wb.save('../data/test_data1.xlsx')