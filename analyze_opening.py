import time
import sys
import numpy as np
import gc
import h5py
import os
import datetime
from config import *
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from State import RIGHT, DOWN, State, BOARD_LEN
import pickle
import argparse
from BasicAI import display_parameter
import pandas as pd
from pprint import pprint
from util import Official2Glendenning, generate_opening_tree, get_normalized_state
from Tree import OpeningTree


if __name__ == "__main__":
    target_opening_data = [  # official notation
        ("d3h, c6h, d5v", ['e2', 'e8', 'e3', 'e7', 'e4', 'e6', 'd3h', 'c6h', 'd5v']),
        ("d3h, c6h, e6v", ['e2', 'e8', 'e3', 'e7', 'e4', 'e6', 'd3h', 'c6h', 'e6v']),
        ("a3h", ['e2', 'e8', 'e3', 'e7', 'e4', 'e6', 'a3h']),
        ("e2, e8, e3, d3h", ['e2', 'e8', 'e3', 'd3h']),
        ("e2, e8, e3, c7h", ['e2', 'c7h', 'e3', 'e8'])
    ]
    target_opening_names = [x[0] for x in target_opening_data]
    target_openings = [x[1] for x in target_opening_data]

    # メモリ使用量に注意。240225時点で、20エポック分で1GBほど消費。
    target_epoch = 4067
    epoch_num_for_joseki = 10

    min_size = 6  # これより探訪数が少ない定石は除外する
    max_depth = 15  # これより深い定跡は作らない（メモリ節約のため）
    max_win_rate_diff = 0.15
    joseki_num = 100
    n_cand = 3
    #n_div_try = 10

    #target_dir = KIFU_DIR
    target_dir = os.path.join("other_records", "240423")
    save_dir = os.path.join("application_data", "joseki")
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = target_epoch - epoch_num_for_joseki
    end_epoch = target_epoch

    all_kifu_list = []

    for epoch in range(start_epoch, end_epoch):
        files = os.listdir(os.path.join(target_dir, str(epoch)))
        files = [file for file in files if file.endswith(".txt")]
        for file in files:
            with open(os.path.join(target_dir, str(epoch), file)) as fin:
                kifu_text = fin.read()
                kifu_list = kifu_text.splitlines()
                kifu_list = [text.strip().split(",") for text in kifu_list]

                all_kifu_list.extend(kifu_list)

    opening_tree, statevec2node = generate_opening_tree(target_epoch, all_kifu_list, max_depth)

    for target_opening in target_openings:
        state, state_vec, _ = get_normalized_state(list(map(Official2Glendenning, target_opening)))
        tree = statevec2node[state_vec]
        print("{} ({:.2f}%), p1 win rate = {:.2f}%".format(tree.visited_num, tree.visited_num / tree.game_num * 100, tree.p1_win_num / tree.visited_num * 100))

