# coding:utf-8
import numpy as np

epsilon = 1e-7
dummy_num = 2  # ダミーで追加する引き分け試合数


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# この関数を最小化する。ただし凸関数とする。
def obj_f(r, n, r_arr, n_arr):
    return -np.sum(n_arr * np.log(sigmoid(r - r_arr)) + (n - n_arr) * np.log(sigmoid(r_arr - r)))


def obj_f_grad(r, n, r_arr, n_arr):
    return -np.sum(n_arr * sigmoid(r_arr - r) - (n - n_arr) * sigmoid(r - r_arr))


# a, bの間に最小値があることを前提とする
def binary_search(a, b, n, r_arr, n_arr):
    c = (a + b) / 2
    if b - a < epsilon:
        return c

    grad = obj_f_grad(c, n, r_arr, n_arr)
    if grad > 0:
        return binary_search(a, c, n, r_arr, n_arr)
    else:
        return binary_search(c, b, n, r_arr, n_arr)


def minimize(n, r_arr, n_arr):
    a = -1
    b = 1
    while obj_f_grad(a, n, r_arr, n_arr) >= 0:
        a *= 2
    while obj_f_grad(b, n, r_arr, n_arr) <= 0:
        b *= 2
    return binary_search(a, b, n, r_arr, n_arr)


def calc_rate(n, r_arr, n_arr):
    n += dummy_num
    n_arr += dummy_num / 2
    return minimize(n, r_arr, n_arr)


if __name__ == "__main__":
    n = 10
    r_arr = np.array([0.0, 1.0])
    n_arr = np.array([8.0, 6.0])
    print(calc_rate(n, r_arr, n_arr))

    n = 100
    r_arr = np.array([0.0, 1.0])
    n_arr = np.array([80.0, 60.0])
    print(calc_rate(n, r_arr, n_arr))

    n = 100
    r_arr = np.array([0.0, 1.0])
    for n1 in [95, 98, 99, 100]:
        for n2 in [95, 98, 99, 100]:
            n_arr = np.array([float(n1), float(n2)])
            print(n1, n2, calc_rate(n, r_arr, n_arr))

