# coding:utf-8
from State import State, BIT_BOARD_LEN, State, State_init, set_state_by_wall, movable_array, accept_action_str, calc_dist_array, placable_array, display_cui
import numpy as np
import os, sys
import ctypes
from BasicAI import state_copy

test_case_list = range(1, 16)
TEST_DIR = "testcases/placable_array"

if os.name == "nt":
    lib = ctypes.CDLL('./State_util.dll')
else:
    lib = ctypes.CDLL('./State_util.so')


set_row_wall_1 = lib.set_row_wall_1
set_row_wall_1.argtypes = [ctypes.POINTER(State), ctypes.c_int, ctypes.c_int]
set_row_wall_1.restype = None
set_row_wall_0 = lib.set_row_wall_0
set_row_wall_0.argtypes = [ctypes.POINTER(State), ctypes.c_int, ctypes.c_int]
set_row_wall_0.restype = None

set_column_wall_1 = lib.set_column_wall_1
set_column_wall_1.argtypes = [ctypes.POINTER(State), ctypes.c_int, ctypes.c_int]
set_column_wall_1.restype = None
set_column_wall_0 = lib.set_column_wall_0
set_column_wall_0.argtypes = [ctypes.POINTER(State), ctypes.c_int, ctypes.c_int]
set_column_wall_0.restype = None

print_state = lib.print_state
print_state.argtypes = [ctypes.POINTER(State)]
print_state.restype = None

eq_state = lib.eq_state
eq_state.argtypes = [ctypes.POINTER(State), ctypes.POINTER(State)]
eq_state.restype = ctypes.c_bool

test_search_util = lib.test_search_util
test_search_util.argtypes = []
test_search_util.restype = None

test_search_util()

results = []
for i in test_case_list:
    state = State()
    State_init(state)
    row_wall = np.loadtxt(os.path.join(TEST_DIR, "{}/r.txt".format(i)), delimiter=",").T
    column_wall = np.loadtxt(os.path.join(TEST_DIR, "{}/c.txt".format(i)), delimiter=",").T
    
    if os.path.exists(os.path.join(TEST_DIR, "{}/p.txt".format(i))):
        p = np.loadtxt(os.path.join(TEST_DIR, "{}/p.txt".format(i)), delimiter=",")
        p = np.asarray(p, dtype=int)
        state.Bx = p[0]
        state.By = p[1]
        state.Wx = p[2]
        state.Wy = p[3]
    set_state_by_wall(state, row_wall, column_wall)
    print_state(state)
    display_cui(state)

    movable_array_ans = np.array(np.loadtxt(os.path.join(TEST_DIR, f"{i}/movable_array.csv"), delimiter=","), dtype=bool)
    movable_array_pred = movable_array(state, state.Bx, state.By)
    if not np.all(movable_array_pred == movable_array_ans):
        print("movable array failed")
        print("ans")
        print(movable_array_ans)
        print("pred")
        print(movable_array_pred)

    placabler_ans = np.array(np.loadtxt(os.path.join(TEST_DIR, f"{i}/placabler.csv"), delimiter=","), dtype=int)
    placablec_ans = np.array(np.loadtxt(os.path.join(TEST_DIR, f"{i}/placablec.csv"), delimiter=","), dtype=int)

    placabler_pred, placablec_pred = placable_array(state, 0)

    if not np.all(placabler_pred == placabler_ans) or not np.all(placablec_pred == placablec_ans):
        print("placable array failed")
        print("row answer")
        print(placabler_ans)
        print("row pred")
        print(placabler_pred)
        print("column answer")
        print(placablec_ans)
        print("column pred")
        print(placablec_pred)
        assert False

    if os.path.exists(os.path.join(TEST_DIR, f"{i}/dist_0.csv")):
        dist_0 = np.loadtxt(os.path.join(TEST_DIR, f"{i}/dist_0.csv"), delimiter=",")
        dist_8 = np.loadtxt(os.path.join(TEST_DIR, f"{i}/dist_8.csv"), delimiter=",")
        dist_0_pred = calc_dist_array(state, 0)
        dist_8_pred = calc_dist_array(state, 8)
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
    sys.stdout.flush()
