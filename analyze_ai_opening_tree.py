import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # warning抑制
import tensorflow as tf
# tf.disable_v2_behavior()
import numpy as np
import random
import matplotlib.pyplot as plt
from config import *
from calc_multi_rate import estimate_multi_rate
from collections import Counter
import copy
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import logging
from util import load_statevec2node, get_epoch_dir_name, generate_opening_tree, save_tree_graph, compute_contributions, update_opening_tree_with_new_kifu, MCTS_select, remove_nodes_below_threshold, select_and_get_nodess_and_actionss, get_normalized_action_list, mirror_action, update_array_with_beta, display_parameter, get_flipped_index
tf.get_logger().setLevel(logging.ERROR)
from State import State, accept_action_str, State_init
from Tree import Glendenning2Official, load_dict_to_opening_tree
from Agent import actionid2str, str2actionid


if __name__ == "__main__":
    load_dir = os.path.join(AI_JOSEKI_DIR, "250213")
    OPENING_NAME = "opening_tree_cycle_750"
    THRESHOLD = 1
    load_path = os.path.join(load_dir, f"{OPENING_NAME}.json")

    with open(load_path, "r") as fin:
        json_dict = json.load(fin)
    opening_tree = load_dict_to_opening_tree(json_dict)
    print("load_dict_to_opening_tree done")
    statevec2node = load_statevec2node(opening_tree)
    print("load_statevec2node done")

    remove_nodes_below_threshold(opening_tree, statevec2node, threshold=THRESHOLD)
    print("remove_nodes_below_threshold done")

    #save_tree_graph(opening_tree, statevec2node, os.path.join(load_dir, "opening_tree_graph_trim"))
    with open(os.path.join(load_dir, f"{OPENING_NAME}_th{THRESHOLD}.json"), "w") as fout:
        json.dump(opening_tree.to_dict(), fout)
