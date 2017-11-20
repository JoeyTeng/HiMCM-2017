# @Author: Joey Teng <Toujour>
# @Date:   19-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: Trail.py
# @Last modified by:   Toujour
# @Last modified time: 19-Nov-2017

import json
import sys

import matplotlib.pyplot
import numpy
from mpl_toolkits.mplot3d import Axes3D

DefaultColor = 0
BackgroundColor = 1
Height = 200
Step = 80

origin_height = int(sys.argv[3])
target_height = int(sys.argv[4])


def line(origin, target):
    # print(origin, target)
    x = numpy.linspace(origin[0], target[0], num=Step, endpoint=True)
    y = numpy.linspace(origin[1], target[1], num=Step, endpoint=True)
    z = numpy.linspace(origin[2], target[2], num=Step, endpoint=True)

    return (x.tolist(), y.tolist(), z.tolist(),)


def pair(origin_img, target_img):
    p = []
    updated = True
    while updated:
        updated = False
        prev = None
        for i in range(origin_img.shape[0]):
            for j in range(origin_img.shape[1]):
                if origin_img[i][j][origin_height] > 0:
                    prev = (i, j, origin_height)
                    origin_img[i][j][origin_height] -= 1
                    break
            if prev is not None:
                break

        for i in range(target_img.shape[0]):
            for j in range(target_img.shape[1]):
                if target_img[i][j][target_height] > 0:
                    p.append(((prev[0], prev[1], prev[2]),
                              (i, j, target_height), ))

                    target_img[i][j][target_height] -= 1
                    updated = True
                    break
            if updated:
                break

    return p


print("INFO: Start", flush=True)

fig = matplotlib.pyplot.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)


origin_vertices = json.load(open(sys.argv[1], 'r'))
target_vertices = json.load(open(sys.argv[2], 'r'))
print("INFO: Vertices loaded", flush=True)

x_max = 0
y_max = 0
for i in range(len(origin_vertices)):
    origin_vertices[i][0] = int(origin_vertices[i][0])
    origin_vertices[i][1] = int(origin_vertices[i][1])
for i in range(len(target_vertices)):
    target_vertices[i][0] = int(target_vertices[i][0])
    target_vertices[i][1] = int(target_vertices[i][1])
for x, y in origin_vertices:
    x_max = max(x_max, x)
    y_max = max(y_max, y)
for x, y in target_vertices:
    x_max = max(x_max, x)
    y_max = max(y_max, y)

origin = numpy.ones((x_max + 1, y_max + 1, max(origin_height, target_height) +
                     1), dtype='int64') * BackgroundColor
target = numpy.ones(origin.shape) * BackgroundColor

assert(len(origin_vertices) == len(target_vertices))

for vertex in origin_vertices:
    origin[vertex[0]][vertex[1]][origin_height] += 1
for vertex in target_vertices:
    target[vertex[0]][vertex[1]][target_height] += 1

print("INFO: Intermediate graph plotted", flush=True)

assert(len(origin) == len(target))

pairs = pair(origin, target)
print("INFO: Pairs found", flush=True)

x = []
y = []
z = []
xn = []
yn = []
zn = []
for pair in pairs:
    xx, yy, zz = line(pair[0], pair[1])
    x.extend(xx)
    y.extend(yy)
    z.extend(zz)
    xn.append(xx)
    yn.append(yy)
    zn.append(zz)

print("INFO: Trail Designed", flush=True)

for i in range(Step):
    ax = Axes3D(fig)
    ax.axis([0, x_max, 0, y_max])
    ax.set_zlim(min(
        origin_height, target_height), max(origin_height, target_height))
    for j in range(len(xn)):
        ax.scatter(xn[j][i], yn[j][i], zn[j][i], c='r', marker='.')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=45., azim=315.)
    matplotlib.pyplot.savefig("movie%d.png" % i)
    matplotlib.pyplot.cla()

# ax.scatter(xt, yt, zt, c='b', marker='^')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# matplotlib.pyplot.show()
# for ii in range(0, 360, 1):
#    ax.view_init(elev=10., azim=ii)
#    matplotlib.pyplot.savefig("movie%d.png" % ii)

# fig.savefig('tmp.png')

print("INFO: Completed", flush=True)
