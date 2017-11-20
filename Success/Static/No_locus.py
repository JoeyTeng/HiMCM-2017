# @Author: Joey Teng <Toujour>
# @Date:   19-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: Polar2Cart_wheel.py
# @Last modified by:   Toujour
# @Last modified time: 19-Nov-2017


import json
import sys

import matplotlib.pyplot
import numpy
from mpl_toolkits.mplot3d import Axes3D

# orbit = numpy.array("(x, +-200/250 (250^2-x^2)^0.5, 0)")
Step = 72
Focus = (0, 0, 0)
R_star = 24


def loadCSV(filename):
    return numpy.array(list(map(lambda x: tuple(map(float, x.split(','))), open(filename, 'r').read().split('\n')))).transpose()


def orbit(x_array):
    y = []
    for x in x_array:
        y.append(200 / 250 * (250**2 - x**2)**0.5)

    return numpy.array(y)


def shift(planet, r=12):
    x0 = numpy.cos(numpy.linspace(
        0, numpy.pi, num=(Step >> 1), endpoint=False)) * 250
    orbit0 = numpy.array(orbit(x0).tolist() + (orbit(x0) * (-1)).tolist())
    print(orbit0)
    x0 = numpy.cos(numpy.linspace(
        0, 2 * numpy.pi, num=Step, endpoint=False)) * 250
    # z0 = numpy.zeros(Step)
    z0 = x0
    drones = []

    for coord in planet:
        x = x0 + coord[0] * r
        y = orbit0 + coord[1] * r
        z = z0 + coord[2] * r
        cart = []

        for i in range(x.shape[0]):
            cart.append((x[i], y[i], z[i]))

        drones.append(cart)

    drones = numpy.array(drones)
    print(drones.shape)
    locus = numpy.zeros([drones.shape[1], drones.shape[2], drones.shape[0]])
    print(locus.shape)
    for drone in range(drones.shape[0]):
        for axis in range(drones.shape[2]):
            for frame in range(drones.shape[1]):
                try:
                    locus[frame][axis][drone] = drones[drone][frame][axis]
                except IndexError:
                    print("{0} {1} {2}".format(frame, axis, drone), flush=True)
                    assert(False)

    return numpy.array(locus).tolist()


def animate(locus, height, x, y, z):
    print("INFO: Start Animation", flush=True)

    fig = matplotlib.pyplot.figure()

    for i in range(1):
        ax = Axes3D(fig)
        mi = min([x[0], y[0], z[0]])
        ma = max([x[1], y[1], z[1]])
        ax.axis([mi, ma, mi, ma])
        ax.set_zlim(mi, ma)
        ax.scatter(locus[i][0], locus[i][1], locus[i][2], c='r', marker='.')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.view_init(elev=10., azim=330.)
        matplotlib.pyplot.savefig("movie%d.png" % i)
        matplotlib.pyplot.cla()

    print("INFO: Animated!", flush=True)


if __name__ == '__main__':
    print("INFO: Start", flush=True)
    star = numpy.array(loadCSV(sys.argv[1])).transpose()

    print(numpy.array(star).shape)
    locus = numpy.zeros([1, star.shape[0], star.shape[1]]).tolist()
    for frame in locus:
        for i in range(3):  # Axis
            frame[i].extend(
                (numpy.array(star[i]) * R_star + Focus[i]).tolist())

    x = 1000
    y = 1000
    z = 1000
    x_m = -1000
    y_m = -1000
    z_m = -1000
    for frame in locus:
        for drone in range(len(frame[0])):
            x = min(x, frame[0][drone])
            y = min(y, frame[1][drone])
            z = min(z, frame[2][drone])
            x_m = max(x_m, frame[0][drone])
            y_m = max(y_m, frame[1][drone])
            z_m = max(z_m, frame[2][drone])

    for i in range(len(locus)):
        for j in range(len(locus[i][0])):
            locus[i][0][j] -= x
            locus[i][1][j] -= y
            locus[i][2][j] -= z

    animate(locus, int(sys.argv[2]), (0, x_m - x), (0, y_m - y), (0, z_m - z))
