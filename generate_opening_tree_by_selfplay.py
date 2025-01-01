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
from util import get_epoch_dir_name, generate_opening_tree, save_tree_graph, compute_contributions, update_opening_tree_with_new_kifu
tf.get_logger().setLevel(logging.ERROR)

from extract_good_AIs import evaluate_2game_process_2id

def selfplay_cycle(
    opening_tree,       # <--- 変更: 既に作成済みのOpeningTreeを受け取る
    statevec2node,      # <--- 同上
    ai_param,
    process_num=4,
    game_num_per_process=2,
    max_depth=20
):
    """
    1サイクル:
      1) マルチプロセスで ai_param vs ai_param の自己対戦を合計8局行う
      2) 得られた棋譜(=new_kifu_list)を元に、OpeningTreeを差分更新
      3) 更新後のOpeningTreeと全棋譜リストを返す
    """
    # 1) 自己対戦
    args_list = []
    for idx in range(process_num):
        seed_val = random.randint(0, 10**6)
        args = (
            0, 0, ai_param, ai_param,
            [], [], [], [],
            seed_val,
            0.0
        )
        args_list.append(args)

    results = []
    with Pool(processes=process_num) as p:
        imap_ret = p.imap(evaluate_2game_process_2id, iterable=args_list)
        for ret in tqdm(imap_ret, total=process_num, desc="SelfPlay"):
            results.append(ret)

    # 2) 棋譜リスト取得
    new_kifu_list = []
    for (eval_result, _, _) in results:
        (_, _, _, kifu_list) = eval_result
        new_kifu_list.extend(kifu_list)

    # 3) OpeningTreeを差分更新
    opening_tree, statevec2node = update_opening_tree_with_new_kifu(
        opening_tree,
        statevec2node,
        new_kifu_list,
        max_depth=max_depth
    )

    return opening_tree, statevec2node, new_kifu_list


def main():
    ai_param = {
        'config_id': 0,
        'AI_id': 15000,
        'search_nodes': 1000,
        'C_puct': 2.5,
        'tau': 0.32,
        'p_tau': 0.7,
        'post_alpha': 2.0,
        'post_beta': 5.0,
        'use_recent_move_vec': True
    }

    PROCESS_NUM = 4
    GAME_NUM_PER_PROCESS = 2
    CYCLE_NUM = 3
    MAX_DEPTH = 20

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
            game_num_per_process=GAME_NUM_PER_PROCESS,
            max_depth=MAX_DEPTH
        )

        all_kifu_list_global.extend(new_kifu_list)

        # 確認用
        print(f"  Cycle {cycle_id+1}: OpeningTree root.visited_num = {opening_tree.visited_num}")

    print("\nAll cycles finished.")
    print(f"Total kifu count: {len(all_kifu_list_global)}")

    os.makedirs(AI_JOSEKI_DIR, exist_ok=True)

    save_tree_graph(opening_tree, statevec2node, os.path.join(AI_JOSEKI_DIR, "opening_tree_graph"))
    with open(os.path.join(AI_JOSEKI_DIR, "opening_tree.json"), "w") as fout:
        json.dump(opening_tree.to_dict(), fout)

if __name__ == "__main__":
    main()
