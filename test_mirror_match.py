from State import State, State_init, State_c, set_state_by_wall, display_cui
import numpy as np
import os
from config import *
import pandas as pd
import time
import ctypes

TEST_DIR = "testcases/mirror_match"  # 勝敗を

dirs = list(os.listdir(TEST_DIR))
dirs = [dir for dir in dirs if dir.isdigit()]
dirs = sorted(dirs, key=int)

start_time = time.time()

if os.name == "nt":
    lib = ctypes.CDLL('./State_util.dll')
else:
    lib = ctypes.CDLL('./State_util.so')
is_mirror_match = lib.is_mirror_match
is_mirror_match.argtypes = [ctypes.POINTER(State_c)]
is_mirror_match.restype = ctypes.c_bool

results = []
for dir in dirs:
    path = os.path.join(TEST_DIR, dir)
    turn = int(os.listdir(path)[0].split("_")[0])
    pos_arr = np.loadtxt(os.path.join(path, f"{turn}_pos.txt"))
    row_wall = np.loadtxt(os.path.join(path, f"{turn}_r.txt"), dtype=bool)
    column_wall = np.loadtxt(os.path.join(path, f"{turn}_c.txt"), dtype=bool)
    p1_walls, p2_walls = np.loadtxt(os.path.join(path, f"{turn}_w.txt"), dtype=int)

    state = State()
    State_init(state)
    state.turn = state.state_c.turn = turn
    state.Bx = state.state_c.Bx = int(pos_arr[0])
    state.By = state.state_c.By = int(pos_arr[1])
    state.Wx = state.state_c.Wx = int(pos_arr[2])
    state.Wy = state.state_c.Wy = int(pos_arr[3])
    state.black_walls = state.state_c.black_walls = p1_walls
    state.white_walls = state.state_c.white_walls = p2_walls
    set_state_by_wall(state, row_wall, column_wall)

    display_cui(state)
    print(is_mirror_match(state.state_c))
    
    results.append(is_mirror_match(state.state_c))

answer_df = pd.read_csv(os.path.join(TEST_DIR, "answer.csv"))
print(answer_df)
expected_result_arr = answer_df["expected result"].values
expected_result_arr = expected_result_arr[:len(results)]
results_arr = np.array(results)
print("correct: ", np.array(results_arr == expected_result_arr, dtype=int))
if np.all(results_arr == expected_result_arr):
    print("passed all testcases")
else:
    print("failed")
print("elapsed time = {:.2f}s".format(time.time() - start_time))
