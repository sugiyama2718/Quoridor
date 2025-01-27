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
from util import get_epoch_dir_name, generate_opening_tree, save_tree_graph, compute_contributions, update_opening_tree_with_new_kifu, MCTS_select, remove_nodes_below_threshold, select_and_get_nodess_and_actionss, get_normalized_action_list, mirror_action, update_array_with_beta, display_parameter, get_flipped_index
tf.get_logger().setLevel(logging.ERROR)
from State import State, accept_action_str, State_init, feature_int
from Tree import Glendenning2Official
from Agent import actionid2str, str2actionid

from extract_good_AIs import evaluate_2game_process_2id

def selfplay_cycle(
    opening_tree,
    statevec2node,
    ai_param,
    process_num=4,
    max_depth=20
):
    """
    1サイクル:
      1) マルチプロセスで ai_param vs ai_param の自己対戦を合計8局行う
      2) 得られた棋譜(=new_kifu_list)を元に、OpeningTreeを差分更新
      3) 更新後のOpeningTreeと全棋譜リストを返す
    """
    game_num = process_num * 2
    # print(list(opening_tree.tree_c.contents.N_arr))
    # print(list(opening_tree.tree_c.contents.Q_arr))
    # print(opening_tree.P)
    if opening_tree.P is None:
        actionss = [[]] * game_num  # 初期局面の場合のみ、空リストで埋める
        nodess = [None] * game_num
    else:
        nodess, actionss = select_and_get_nodess_and_actionss(opening_tree, 10.0, 0, 0, game_num, game_num, 1)

    new_actionss = []
    for nodes, actions in zip(nodess, actionss):
        s = State()
        mirror_s = State()
        State_init(s)
        State_init(mirror_s)

        is_success = True
        prev_is_mirrored = False

        new_actions = []

        for action in actions:
            if prev_is_mirrored:
                action = get_flipped_index(action)
            action_str = actionid2str(s, action)
            new_actions.append(action)
            
            mirror_action = get_flipped_index(action)
            mirror_action_str = actionid2str(mirror_s, mirror_action)

            is_success = is_success and accept_action_str(s, action_str)
            is_success = is_success and accept_action_str(mirror_s, mirror_action_str)

            state_vec = tuple(feature_int(s).flatten())
            mirror_state_vec = tuple(feature_int(mirror_s).flatten())

            is_mirrored = (state_vec > mirror_state_vec)

            official_str = Glendenning2Official(action_str)
            print(official_str, end=", ")
            #print(official_str, end=", ")

            prev_is_mirrored = is_mirrored
        new_actionss.append(new_actions)

        print()
        if not is_success:
            print("contains illegal move!!!")
            if nodes is not None:
                for node, action in zip(nodes, actions):
                    print("-"*30)
                    print(action)
                    print(display_parameter(np.array(node.P * 1000, dtype=int)))
                    print(display_parameter(np.array(node.tree_c.contents.N_arr, dtype=int)))
                    print(display_parameter(np.array(np.array(node.tree_c.contents.Q_arr) * 1000, dtype=int)))

    actionss = new_actionss

    # 1) 自己対戦
    args_list = []
    for idx in range(process_num):
        seed_val = random.randint(0, 10**6)
        args = (
            0, 0, ai_param, ai_param,
            [], [], [], [],
            seed_val,
            0.0,
            actionss[idx*2:idx*2+2]
        )
        args_list.append(args)

    results = []
    with Pool(processes=process_num) as p:
        imap_ret = p.imap(evaluate_2game_process_2id, iterable=args_list)
        for ret in tqdm(imap_ret, total=process_num, desc="SelfPlay"):
            results.append(ret)

    # 2) 棋譜リスト取得
    new_kifu_list = []
    pi_lists = []
    for (eval_result, _, _) in results:
        (_, _, _, kifu_list, pi_list) = eval_result
        new_kifu_list.extend(kifu_list)
        pi_lists.extend(pi_list)

    # 3) OpeningTreeを差分更新
    opening_tree, statevec2node = update_opening_tree_with_new_kifu(
        opening_tree,
        statevec2node,
        new_kifu_list,
        max_depth=max_depth,
        pi_lists=pi_lists
    )

    return opening_tree, statevec2node, new_kifu_list


def main():
    ai_param = {
        'config_id': 0,
        'AI_id': 15000,
        'search_nodes': 500,
        'C_puct': 2.5,
        #'tau': 0.32,
        'tau': 0.64,
        'p_tau': 0.7,
        'post_alpha': 2.0,
        'post_beta': 5.0,
        'use_recent_move_vec': True
    }

    PROCESS_NUM = 4
    CYCLE_NUM = 50
    MAX_DEPTH = 200  # AI向け定石なので、必要があればいくらでも深く探索させたい

    # --- 初回だけ generate_opening_tree(空リストで良いなら空でOK) ---
    opening_tree, statevec2node = generate_opening_tree(
        all_kifu_list=[],
        max_depth=MAX_DEPTH
    )

    all_kifu_list_global = []

    for cycle_id in range(CYCLE_NUM):
        print(f"\n===== Cycle {cycle_id+1} start =====")

        opening_tree, statevec2node, new_kifu_list = selfplay_cycle(
            opening_tree,  # 既存のツリーを渡す
            statevec2node,
            ai_param,
            process_num=PROCESS_NUM,
            max_depth=MAX_DEPTH
        )

        all_kifu_list_global.extend(new_kifu_list)

        # 確認用
        print(f"  Cycle {cycle_id+1}: OpeningTree root.visited_num = {opening_tree.visited_num}")

    print("\nAll cycles finished.")
    print(f"Total kifu count: {len(all_kifu_list_global)}")

    remove_nodes_below_threshold(opening_tree, statevec2node)

    os.makedirs(AI_JOSEKI_DIR, exist_ok=True)

    save_tree_graph(opening_tree, statevec2node, os.path.join(AI_JOSEKI_DIR, "opening_tree_graph"))
    with open(AI_OPENING_TREE_DEFAULT_PATH, "w") as fout:
        json.dump(opening_tree.to_dict(), fout)

if __name__ == "__main__":
    main()
