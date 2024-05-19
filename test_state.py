# coding:utf-8
from State import State, BIT_BOARD_LEN, State_c, State_init
import numpy as np
import os
import ctypes
from BasicAI import state_copy

test_case_list = range(1, 16)
TEST_DIR = "testcases/placable_array"

if os.name == "nt":
    lib = ctypes.CDLL('./State_util.dll')
else:
    lib = ctypes.CDLL('./State_util.so')


set_row_wall_1 = lib.set_row_wall_1
set_row_wall_1.argtypes = [ctypes.POINTER(State_c), ctypes.c_int, ctypes.c_int]
set_row_wall_1.restype = None
set_row_wall_0 = lib.set_row_wall_0
set_row_wall_0.argtypes = [ctypes.POINTER(State_c), ctypes.c_int, ctypes.c_int]
set_row_wall_0.restype = None

set_column_wall_1 = lib.set_column_wall_1
set_column_wall_1.argtypes = [ctypes.POINTER(State_c), ctypes.c_int, ctypes.c_int]
set_column_wall_1.restype = None
set_column_wall_0 = lib.set_column_wall_0
set_column_wall_0.argtypes = [ctypes.POINTER(State_c), ctypes.c_int, ctypes.c_int]
set_column_wall_0.restype = None

print_state = lib.print_state
print_state.argtypes = [ctypes.POINTER(State_c)]
print_state.restype = None

results = []
for i in test_case_list:
    state = State()
    State_init(state)
    state.row_wall = np.loadtxt(os.path.join(TEST_DIR, "{}/r.txt".format(i)), delimiter=",").T
    state.column_wall = np.loadtxt(os.path.join(TEST_DIR, "{}/c.txt".format(i)), delimiter=",").T
    state.set_state_by_wall()
    if os.path.exists(os.path.join(TEST_DIR, "{}/p.txt".format(i))):
        p = np.loadtxt(os.path.join(TEST_DIR, "{}/p.txt".format(i)), delimiter=",")
        p = np.asarray(p, dtype="int8")
        state.Bx = state.state_c.Bx = p[0]
        state.By = state.state_c.By = p[1]
        state.Wx = state.state_c.Wx = p[2]
        state.Wy = state.state_c.Wy = p[3]
    print_state(state.state_c)
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

