import sys
sys.path.append("./")
from BasicAI import calc_optimal_move_by_DP
from State import State, State_init
from main import normal_play
import numpy as np
import os
from CNNAI import CNNAI
from config import *
import pandas as pd
import time
from util import get_epoch_dir_name

#TEST_DIR = "endgame_testcase/240119"  # 確定路のテスト
TEST_DIR = "endgame_testcase/240128"  # 勝敗を

dirs = list(os.listdir(TEST_DIR))
dirs = [dir for dir in dirs if dir.isdigit()]
dirs = sorted(dirs, key=int)

TARGET_EPOCH = 2910
SEARCH_NODES = 1000

AIs = [CNNAI(0, search_nodes=SEARCH_NODES, tau=0.0, seed=100), CNNAI(1, search_nodes=SEARCH_NODES, tau=0.0, seed=200)]
AIs[0].load(os.path.join(PARAMETER_DIR, get_epoch_dir_name(TARGET_EPOCH), f"epoch{TARGET_EPOCH}.ckpt"))
AIs[1].load(os.path.join(PARAMETER_DIR, get_epoch_dir_name(TARGET_EPOCH), f"epoch{TARGET_EPOCH}.ckpt"))

start_time = time.time()

results = []
for dir in dirs:
    for i in range(10):
        print("="*60)
    print(dir)
    path = os.path.join(TEST_DIR, dir)
    turn = int(os.listdir(path)[0].split("_")[0])
    pos_arr = np.loadtxt(os.path.join(path, f"{turn}_pos.txt"))
    row_wall = np.loadtxt(os.path.join(path, f"{turn}_r.txt"), dtype=bool)
    column_wall = np.loadtxt(os.path.join(path, f"{turn}_c.txt"), dtype=bool)
    p1_walls, p2_walls = np.loadtxt(os.path.join(path, f"{turn}_w.txt"))

    state = State()
    State_init(state)
    state.turn = turn
    state.Bx = int(pos_arr[0])
    state.By = int(pos_arr[1])
    state.Wx = int(pos_arr[2])
    state.Wy = int(pos_arr[3])
    state.black_walls = p1_walls
    state.white_walls = p2_walls
    state.set_state_by_wall()

    # 最初のターンによって手番が違うことを考慮
    if turn % 2 == 0:
        AIs[0].color = 0
        AIs[1].color = 1
    else:
        AIs[0].color = 1
        AIs[1].color = 0  

    AIs[0].init_prev()
    AIs[1].init_prev()
    reward = normal_play(AIs, initial_state=state)
    results.append(reward)

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
