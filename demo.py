# @Author: Joey Teng <Toujour>
# @Date:   18-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: demo.py
# @Last modified by:   Toujour
# @Last modified time: 18-Nov-2017


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def partA(x):
    return (25 / 128 * numpy.sqrt(3))


def func(x):
    return 25


a, b = 2, 9  # integral limits
x = np.linspace(0, 10)
y = func(x)

fig, ax = plt.subplots()
plt.plot(x, y, 'r', linewidth=2)
plt.ylim(ymin=0)

# Make the shaded region
ix = np.linspace(a, b)
iy = func(ix)
verts = [(a, 0)] + list(zip(ix, iy)) + [(b, 0)]
poly = Polygon(verts, facecolor='1', edgecolor='0')
ax.add_patch(poly)


plt.savefig('tmp.png')
