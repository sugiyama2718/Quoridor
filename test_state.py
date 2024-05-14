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

    # np.savetxt(os.path.join(TEST_DIR, f"{i}/dist_0.csv"), state.dist_array(0, state.cross_movable_array2(state.row_wall, state.column_wall)),delimiter=",")
    # np.savetxt(os.path.join(TEST_DIR, f"{i}/dist_8.csv"), state.dist_array(8, state.cross_movable_array2(state.row_wall, state.column_wall)),delimiter=",")
    if os.path.exists(os.path.join(TEST_DIR, f"{i}/dist_0.csv")):
        dist_0 = np.loadtxt(os.path.join(TEST_DIR, f"{i}/dist_0.csv"), delimiter=",")
        dist_8 = np.loadtxt(os.path.join(TEST_DIR, f"{i}/dist_8.csv"), delimiter=",")
        dist_0_pred = state.dist_array(0, state.cross_movable_array2(state.row_wall, state.column_wall))
        dist_8_pred = state.dist_array(8, state.cross_movable_array2(state.row_wall, state.column_wall))
        if not np.all(dist_0_pred == dist_0):
            print("dist_0")
            print(dist_0)
            print(dist_0_pred)
            assert False
        if not np.all(dist_8_pred == dist_8):
            print("dist_8")
            print(dist_8)
            print(dist_8_pred)
            assert False

