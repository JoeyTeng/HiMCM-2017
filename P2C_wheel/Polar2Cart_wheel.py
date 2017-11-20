# @Author: Joey Teng <Toujour>
# @Date:   19-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: Polar2Cart_wheel.py
# @Last modified by:   Toujour
# @Last modified time: 19-Nov-2017


import json
import sys

import numpy


def loadCSV(filename):
    return numpy.array(list(map(float, open(filename, 'r').read().split())))
    # return open(filename, 'r').read().split()


def loadCSV2(filename):
    return list(map(lambda x: tuple(map(float, x.split(','))), open(filename, 'r').read().split('\n')))


def polar2Cart(angle, r=75):
    x = numpy.cos(angle / 180 * numpy.pi) * r
    y = numpy.sin(angle / 180 * numpy.pi) * r
    cart = []

    for i in range(x.shape[0]):
        cart.append((x[i], y[i],))

    return cart


if __name__ == '__main__':
    angle = loadCSV(sys.argv[1])
    wheel = polar2Cart(angle)
    spoke = []
    for r in numpy.linspace(12.5, 75, num=5, endpoint=False):
        spoke.extend(polar2Cart(numpy.linspace(
            2 * numpy.pi, 0, num=17, endpoint=False), r=r))
    triangle = loadCSV2(sys.argv[2])

    ferris = wheel + triangle + spoke + [(0, 0)]

    x = 0
    y = 0
    for ix, iy in ferris:
        x = min(x, ix)
        y = min(y, iy)

    ferris = numpy.array(ferris)
    for i in range(ferris.shape[0]):
        ferris[i][0] -= x
        ferris[i][1] -= y

    ferris = ferris.tolist()
    json.dump(ferris, open(sys.argv[3], 'w'))
