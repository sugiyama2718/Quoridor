# coding:utf-8
from State import State, BIT_BOARD_LEN
import numpy as np
import os

test_case_list = range(1, 16)
TEST_DIR = "testcases/placable_array"

results = []
for i in test_case_list:
    state = State()
    state.row_wall = np.loadtxt(os.path.join(TEST_DIR, "{}/r.txt".format(i)), delimiter=",").T
    state.column_wall = np.loadtxt(os.path.join(TEST_DIR, "{}/c.txt".format(i)), delimiter=",").T
    for x in range(8):
        for y in range(8):
            if state.row_wall[x, y]:
                state.row_wall_bit[x + y * BIT_BOARD_LEN] = 1
            if state.column_wall[x, y]:
                state.column_wall_bit[x + y * BIT_BOARD_LEN] = 1  
    if os.path.exists(os.path.join(TEST_DIR, "{}/p.txt".format(i))):
        p = np.loadtxt(os.path.join(TEST_DIR, "{}/p.txt".format(i)), delimiter=",")
        p = np.asarray(p, dtype="int8")
        state.Bx = p[0]
        state.By = p[1]
        state.Wx = p[2]
        state.Wy = p[3]
    state.display_cui(check_algo=True)


