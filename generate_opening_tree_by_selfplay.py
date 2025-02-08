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
from util import (
    get_epoch_dir_name, generate_opening_tree, save_tree_graph,
    compute_contributions, update_opening_tree_with_new_kifu, MCTS_select,
    remove_nodes_below_threshold, select_and_get_nodess_and_actionss,
    get_normalized_action_list, mirror_action, update_array_with_beta,
    display_parameter, get_flipped_index, load_statevec2node, set_statevec2node
)
tf.get_logger().setLevel(logging.ERROR)
from State import State, accept_action_str, State_init, feature_int
from Tree import Glendenning2Official, load_dict_to_opening_tree
from Agent import actionid2str, str2actionid
from extract_good_AIs import evaluate_2game_process_2id

import json  # JSON入出力用


# --- selfplay_cycle() ---
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
    estimated_V = 0.0
    if opening_tree.P is None:
        actionss = [[]] * game_num  # 初期局面の場合のみ、空リストで埋める
        nodess = [None] * game_num
    else:
        nodess, actionss = select_and_get_nodess_and_actionss(opening_tree, 10.0, estimated_V, 0, game_num, game_num, 1)

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

    # 1) 自己対戦の準備
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
        pi_lists=pi_lists,
        estimated_V=estimated_V
    )

    return opening_tree, statevec2node, new_kifu_list

# --- main() ---
def main():
    ai_param = {
        'config_id': 0,
        'AI_id': 15000,
        'search_nodes': 1000,
        'C_puct': 2.5,
        'tau': 0.32,
        #'tau': 0.72,
        'p_tau': 0.7,
        'post_alpha': 2.0,
        'post_beta': 5.0,
        'use_recent_move_vec': True,
        'opening_tree_path': None
    }

    PROCESS_NUM = 4
    CYCLE_NUM = 250
    MAX_DEPTH = 200  # AI向け定石なので、必要があればいくらでも深く探索させたい

    # --- 既存のopening_treeがあればjsonから読み込み、なければ初期生成 ---
    if os.path.exists(AI_OPENING_TREE_DEFAULT_PATH):
        print("既存の定石木のjsonが見つかったため、読み込みを行います。")
        with open(AI_OPENING_TREE_DEFAULT_PATH, "r") as fin:
            tree_dict = json.load(fin)

        opening_tree = load_dict_to_opening_tree(tree_dict)
        # 復元したopening_treeからstatevec2nodeを再構築
        statevec2node = load_statevec2node(opening_tree)
        set_statevec2node(opening_tree, statevec2node)
        completed_cycles = opening_tree.selfplay_epoch if opening_tree.selfplay_epoch is not None else 0
        if completed_cycles >= CYCLE_NUM:
            print("定石作成は既に完了しています。")
            return
        remaining_cycles = CYCLE_NUM - completed_cycles
        print(f"サイクル {completed_cycles} 回分は既に実施済み。残り {remaining_cycles} サイクルを実施します。")
    else:
        # 初回の場合
        opening_tree, statevec2node = generate_opening_tree(
            all_kifu_list=[],
            max_depth=MAX_DEPTH
        )
        completed_cycles = 0

    all_kifu_list_global = []

    # completed_cyclesからCYCLE_NUMまでサイクルを回す
    for cycle_id in range(completed_cycles, CYCLE_NUM):
        print(f"\n===== Cycle {cycle_id+1} start =====")

        opening_tree, statevec2node, new_kifu_list = selfplay_cycle(
            opening_tree,  # 既存のツリーを渡す
            statevec2node,
            ai_param,
            process_num=PROCESS_NUM,
            max_depth=MAX_DEPTH
        )

        all_kifu_list_global.extend(new_kifu_list)
        # サイクル進捗をopening_treeに記録（再開時の判断材料とする）
        opening_tree.selfplay_epoch = cycle_id + 1

        # 確認用
        print(f"  Cycle {cycle_id+1}: OpeningTree root.visited_num = {opening_tree.visited_num}")

        # 定期的に（例：50サイクルごと）コンパクト化と保存を実施
        if (cycle_id + 1) % 50 == 0:
            print("  --- 定期コンパクト化および中間保存を実施中 ---")
            remove_nodes_below_threshold(opening_tree, statevec2node)
            # 中間保存ファイル（保存処理に時間がかかるため、頻繁にならないようにremove_nodes_below_thresholdと同期）
            intermediate_path = os.path.join(AI_JOSEKI_DIR, f"opening_tree_cycle_{cycle_id+1}.json")
            with open(intermediate_path, "w") as fout:
                json.dump(opening_tree.to_dict(), fout)
            # メインの定石木jsonも更新
            with open(AI_OPENING_TREE_DEFAULT_PATH, "w") as fout:
                json.dump(opening_tree.to_dict(), fout)

    print("\nAll cycles finished.")
    print(f"Total kifu count: {len(all_kifu_list_global)}")

    # 最終的に一度コンパクト化して保存
    remove_nodes_below_threshold(opening_tree, statevec2node)
    save_tree_graph(opening_tree, statevec2node, os.path.join(AI_JOSEKI_DIR, "opening_tree_graph"))
    with open(AI_OPENING_TREE_DEFAULT_PATH, "w") as fout:
        json.dump(opening_tree.to_dict(), fout)

if __name__ == "__main__":
    main()
