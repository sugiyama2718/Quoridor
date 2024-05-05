# coding:utf-8
from State import State
import numpy as np
import os

#test_case_list = [14]
test_case_list = range(1, 15)

results = []
for i in test_case_list:
    state = State()
    state.row_wall = np.loadtxt("testcase/{}/r.txt".format(i), delimiter=",").T
    state.column_wall = np.loadtxt("testcase/{}/c.txt".format(i), delimiter=",").T
    if os.path.exists("testcase/{}/p.txt".format(i)):
        p = np.loadtxt("testcase/{}/p.txt".format(i), delimiter=",")
        p = np.asarray(p, dtype="int8")
        state.Bx = p[0]
        state.By = p[1]
        state.Wx = p[2]
        state.Wy = p[3]
    state.display_cui()


