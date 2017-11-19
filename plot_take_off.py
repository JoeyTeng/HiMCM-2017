# @Author: Joey Teng <Toujour>
# @Date:   19-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: GraphPlotter.py
# @Last modified by:   Toujour
# @Last modified time: 19-Nov-2017


import json
from sys import argv

import matplotlib.pyplot as plt
import numpy

DefaultColor = '#000000'
BackgroundColor = '#FFFFFF'


def main(path, color=DefaultColor):
    # data: list: [tuple:(x, y)]

    xx = numpy.array(list(range(20)) * 16, dtype='float32') + 0.5
    yy = numpy.array([y for y in range(16)
                      for i in range(20)], dtype='float32') + 0.5

    plt.figure(figsize=(20, 16), dpi=32)
    plt.axis([0, 20, 0, 16])
    plt.scatter(xx, yy, c=DefaultColor, s=32)

    horizontal = numpy.linspace(0, 20)
    vertical = numpy.linspace(0, 16)
    for x in range(20):
        for y in range(16):
            plt.plot(horizontal, numpy.ones(horizontal.shape) * y, c='b')
            plt.plot(numpy.ones(vertical.shape) * x, vertical, c='b')

    plt.savefig(path)
    plt.cla()


if __name__ == '__main__':
    main(argv[1])
