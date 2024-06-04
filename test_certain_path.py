from State import State, State_init, State, movable_array, set_state_by_wall, is_certain_path_terminate, display_cui
import numpy as np
import os
from config import *
import pandas as pd
import time
import ctypes

TEST_DIR = "testcases/certain_path"

dirs = list(os.listdir(TEST_DIR))
dirs = [dir for dir in dirs if dir.isdigit()]
dirs = sorted(dirs, key=int)

start_time = time.time()

if os.name == "nt":
    lib = ctypes.CDLL('./State_util.dll')
else:
    lib = ctypes.CDLL('./State_util.so')
calc_oneside_placable_r_cand_from_color = lib.calc_oneside_placable_r_cand_from_color
calc_oneside_placable_r_cand_from_color.argtypes = [ctypes.POINTER(State), ctypes.c_int]
calc_oneside_placable_r_cand_from_color.restype = (ctypes.c_uint64) * 2
calc_oneside_placable_c_cand_from_color = lib.calc_oneside_placable_c_cand_from_color
calc_oneside_placable_c_cand_from_color.argtypes = [ctypes.POINTER(State), ctypes.c_int]
calc_oneside_placable_c_cand_from_color.restype = (ctypes.c_uint64) * 2

state = State()
State_init(state)
calc_oneside_placable_r_cand_from_color(state, 0)
calc_oneside_placable_c_cand_from_color(state, 0)

results = []
for dir in dirs:
    path = os.path.join(TEST_DIR, dir)
    turn = int(os.listdir(path)[0].split("_")[0])
    pos_arr = np.loadtxt(os.path.join(path, f"{turn}_pos.txt"))
    row_wall = np.loadtxt(os.path.join(path, f"{turn}_r.txt"), dtype=bool)
    column_wall = np.loadtxt(os.path.join(path, f"{turn}_c.txt"), dtype=bool)
    #p1_walls, p2_walls = np.loadtxt(os.path.join(path, f"{turn}_w.txt"), dtype=int)

    state = State()
    State_init(state)
    state.turn = turn
    state.Bx = int(pos_arr[0])
    state.By = int(pos_arr[1])
    state.Wx = int(pos_arr[2])
    state.Wy = int(pos_arr[3])
    set_state_by_wall(state, row_wall, column_wall)

    display_cui(state)

    calc_oneside_placable_r_cand_from_color(state, 0)
    calc_oneside_placable_c_cand_from_color(state, 0)

    B_ans, W_ans = np.array(np.loadtxt(os.path.join(path, f"ans.txt")), dtype=bool)
    B_pred = is_certain_path_terminate(state, 0)
    W_pred = is_certain_path_terminate(state, 1)

    if not (B_pred == B_ans and W_pred == W_ans):
        print("pred")
        print(B_pred, W_pred)
        print("ans")
        print(B_ans, W_ans)
        assert False

