# @Author: Joey Teng <Toujour>
# @Date:   19-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: json2csv.py
# @Last modified by:   Toujour
# @Last modified time: 19-Nov-2017


import json
import sys

import numpy


def CSV(matrix, filename):
    file = open(filename, 'w')
    for i in matrix:
        for j in i:
            file.write('{0},'.format(j))
        file.write('\n')


def loadCSV2(filename):
    return numpy.array(list(map(lambda x: tuple(map(float, x.split(','))), open(filename, 'r').read().split('\n')))).transpose().tolist()


if __name__ == '__main__':
    CSV(loadCSV2(sys.argv[1]), sys.argv[2])
