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
import argparse
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
    target_epoch = 4200
    epoch_num_for_analyze = 900
    each_epoch_num_for_analyze = 10

    max_depth = 15  # これより深い定跡は作らない（メモリ節約のため）

    target_dir = os.path.join("other_records", "240428_all")
    save_dir = os.path.join("application_data", "joseki")
    os.makedirs(save_dir, exist_ok=True)

    start_epoch_all = target_epoch - epoch_num_for_analyze
    end_epoch_all = target_epoch

    visit_rate_dict = {}
    for opening_name in target_opening_names:
        visit_rate_dict[opening_name] = []

    for start_epoch in range(start_epoch_all, end_epoch_all, each_epoch_num_for_analyze):
        end_epoch = start_epoch + each_epoch_num_for_analyze
        print("="*50)
        print(start_epoch, end_epoch)

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

        opening_tree, statevec2node = generate_opening_tree(all_kifu_list, max_depth, target_epoch)

        for opening_name, target_opening in target_opening_data:
            state, state_vec, _ = get_normalized_state(list(map(Official2Glendenning, target_opening)))
            if state_vec in statevec2node.keys():
                tree = statevec2node[state_vec]
                #print("{} ({:.2f}%), p1 win rate = {:.2f}%".format(tree.visited_num, tree.visited_num / tree.game_num * 100, tree.p1_win_num / tree.visited_num * 100))

                visit_rate_dict[opening_name].append(tree.visited_num / tree.game_num)
            else:
                visit_rate_dict[opening_name].append(0.0)

        del opening_tree, statevec2node
        gc.collect()

    visit_rate_df = pd.DataFrame(visit_rate_dict, index=list(range(start_epoch_all, end_epoch_all, each_epoch_num_for_analyze)))
    visit_rate_df.to_csv(os.path.join(save_dir, "visit_rate_df.csv"))

    visit_rate_df.plot()
    plt.savefig(os.path.join(save_dir, "visit_rate.png"))

    print("visit rate csv and png are saved at", save_dir)

