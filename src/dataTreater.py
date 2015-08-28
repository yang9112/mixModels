# -*- coding: utf-8 -*-
import re, os, csv, sys
import xlrd, xlwt
import numpy as np
from xlutils.copy import copy

csv.field_size_limit(sys.maxint)

class DataTreater():
    def __init__(self):
        pass

    def read_csv(self, filename):
        csvfile = file(filename, 'rb')
        reader = csv.reader(csvfile)

        title_data = []
        content_data = []
        
        for line in reader:
            [url, title, content] = line
            title_data.append(title)
            content_data.append(content)
            
        csvfile.close()
        return title_data, content_data

    def read_excel(self, filename):
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

            if len(row_values) > 3 and row_values[3] != '':
                result_data.append(row_values[3])

        return title_data, content_data, result_data

    def read_dict_excel(self, filename, originfile):
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

    def write_data(self, filename_save, filename_data, idx, y_te):
        #get row number
        nrows = 0
        if os.path.isfile(filename_save):
            reader = xlrd.open_workbook(filename_save)
            sheet = reader.sheet_by_index(0)
            nrows = sheet.nrows

            wb = copy(reader)
            sheet = wb.get_sheet(0)
        else:
            wb = xlwt.Workbook()
            sheet = wb.add_sheet(u'sheet1',cell_overwrite_ok=True)

        [content, title, result] = self.read_excel(filename_data)

        content_write = np.array(content)[idx]
        for i in range(len(content_write)):
            sheet.write(i + nrows, 1, content_write[i])
            sheet.write(i + nrows, 2, y_te[i])

        title_write =  np.array(title)[idx]
        for i in range(len(content_write)):
            sheet.write(i + nrows, 0, title_write[i])

        wb.save(filename_save)


if __name__ == '__main__':
    DT = DataTreater()
    [title_data, content_data] = DT.read_csv('../data/test_data.csv')
#    DT.read_dict_excel('../data/dicModelResult.xlsx', '../data/data.xlsx')