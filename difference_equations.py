# @Author: Joey Teng <Toujour>
# @Date:   18-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: demo.py
# @Last modified by:   Toujour
# @Last modified time: 19-Nov-2017
import random

import matplotlib.pyplot as plt
import numpy as np


def v_n_1(v, tol, k=-0.7):
    # -k(v(n+1) - v(n))t=v_{tol_n}
    return (tol / (-k)) + v


def v_n(v, tol, tol_1):
    # v(n)-v(n-1)=v_{tol_n}
    return tol - tol_1 + v


def func(x, tol, v_0):
    v = np.zeros(x.shape)
    v[0] = v_0

    for i in x[1:]:
        v[i] = v_n(v[i - 1], tol[i], tol[i - 1])

    return v


def a_n(v_n, tol, k=-0.7):
    # a=-k(v(n)-vtol(n))
    return (-k) * (v_n - tol)


def funcK(x, tol, v_0):
    # Calculate v
    t = 1
    v = np.zeros(x.shape)
    v[0] = v_0

    for i in x[1:]:
        # v[i + 1] = v_n_1(v[i], tol[i + 1])
        if (i % 4 == 0):
            disturbance = random.randint(-10, 10)
            v[i - 1] += disturbance
            print(disturbance)

        v[i] = v[i - 1] - a_n(v[i - 1], tol[i - 1]) * t

    return v


if __name__ == '__main__':
    v_0 = 10
    x = np.array(list(range(20)))
    tol = np.array(list(list(range(10)) + list(range(9, -1, -1))))
    t = x
    yK = funcK(x, tol, v_0)
    # y = func(x, tol, v_0)

    plt.plot(t, tol, marker='+', color='y', linestyle='-')
    plt.plot(t, yK, marker='o', color='b', linestyle='-')
    # plt.plot(t, y, 'go')
    print(yK)
    plt.axis([0, 20, -10, 20])

    plt.show()
