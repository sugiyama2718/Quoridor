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
from util import Glendenning2Official, generate_opening_tree
from Tree import OpeningTree


if __name__ == "__main__":
    # 親子関係の存在しないノードの組のうち、各ノードの勝率が一定以上50%から離れておらず、できるだけ局面数が多くて各ノードの局面数が近くなるような組を選ぶ
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

    def traverse(tree, actions):
        if tree.visited_num / tree.game_num >= 0.005:
            print(actions)
            print("{} ({:.2f}%), p1 win rate = {:.2f}%".format(tree.visited_num, tree.visited_num / tree.game_num * 100, tree.p1_win_num / tree.visited_num * 100))
            print()
        for key, node in tree.children.items():
            if isinstance(node, OpeningTree):
                traverse(node, actions + [key])

    opening_tree = generate_opening_tree(target_epoch, all_kifu_list, max_depth)

    traverse(opening_tree, [])
