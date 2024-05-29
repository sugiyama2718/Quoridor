from State import State, State_init, State_c, movable_array, set_state_by_wall
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
# is_mirror_match = lib.is_mirror_match
# is_mirror_match.argtypes = [ctypes.POINTER(State_c)]
# is_mirror_match.restype = ctypes.c_bool

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
    state.turn = state.state_c.turn = turn
    state.Bx = state.state_c.Bx = int(pos_arr[0])
    state.By = state.state_c.By = int(pos_arr[1])
    state.Wx = state.state_c.Wx = int(pos_arr[2])
    state.Wy = state.state_c.Wy = int(pos_arr[3])
    state.row_wall = row_wall
    state.column_wall = column_wall
    # state.black_walls = state.state_c.black_walls = p1_walls
    # state.white_walls = state.state_c.white_walls = p2_walls
    set_state_by_wall(state)

    state.display_cui()

    

