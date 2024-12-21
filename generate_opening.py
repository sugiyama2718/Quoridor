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
import heapq


class MaxHeap:
    def __init__(self):
        self.heap = []

    def add(self, element):
        # ヒープに要素を追加（符号を反転させる）
        score, item = element
        heapq.heappush(self.heap, (-score, item))

    def pop(self):
        score, item = heapq.heappop(self.heap)
        return -score, item
    
    def peek(self):
        # 最大の要素を覗き見（符号を元に戻す）
        score, item = self.heap[0]
        return -score, item
    
    def top_n(self, n):
        # ヒープのコピーを作成
        temp_heap = self.heap[:]
        # 上位n個の要素を取り出す
        top_n_elements = [heapq.heappop(temp_heap) for _ in range(min(n, len(self.heap)))]
        top_n_elements = [(-score, item) for score, item in top_n_elements]
        return top_n_elements


if __name__ == "__main__":
    # 親子関係の存在しないノードの組のうち、各ノードの勝率が一定以上50%から離れておらず、できるだけ局面数が多くて各ノードの局面数が近くなるような組を選ぶ
    # メモリ使用量に注意。240225時点で、20エポック分で1GBほど消費。
    target_epoch = 4200
    epoch_num_for_joseki = 100

    min_size = 6  # これより探訪数が少ない定石は除外する
    max_depth = 15  # これより深い定跡は作らない（メモリ節約のため）
    max_win_rate_diff = 0.15
    joseki_num = 100
    n_cand = 3
    #n_div_try = 10

    #target_dir = KIFU_DIR
    target_dir = os.path.join("other_records", "240507_4100kifu")
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

    opening_tree, _ = generate_opening_tree(all_kifu_list, max_depth, target_epoch)

    traverse(opening_tree, [])

    BIG_NUM = 1000

    def node_score(node):
        return float(np.log(node.visited_num / node.game_num) - 0 * abs(node.p1_win_num / node.visited_num - 0.5))
    
    def joseki_score(solution):
        ret = 0.0

        nodes = [x[0] for x in solution]
        action_lists = [x[1] for x in solution]

        for node, action_list in zip(nodes, action_lists):
            ret += node_score(node)

            # 品質確保のため、一つの行動だけからなる定跡はなるべく作られないようにしている。できれば結果的に一つの行動だけからなる定跡が排除されるようなアルゴリズムに変更したい
            if len(action_list) == 1:
                ret -= 10

        return ret + (len(nodes) - joseki_num) * BIG_NUM

    cands = MaxHeap()
    cands.add((node_score(opening_tree), (opening_tree, [])))
    prev_score = -(joseki_num + 1) * BIG_NUM
    count = 0
    best_score = prev_score
    best_solution = cands.top_n(joseki_num)
    determined = []

    while True:
        prev_len = len(cands.heap)
        if prev_len == 0:  # candsは単調に減少するので必ずこの条件で停止する
            break
        node_s, (node, action_list) = cands.pop()
        for action in node.children.keys():
            if isinstance(node.children[action], OpeningTree):
                child = node.children[action]
            else:
                continue
            if child.visited_num < min_size or abs(child.p1_win_num / child.visited_num - 0.5) > max_win_rate_diff:
                continue

            cands.add((node_score(child), (child, action_list + [action])))

        # if len(cands.heap) < prev_len:
        #     determined.append((node_s, (node, action_list)))
        #     determined = determined[:joseki_num]
            # print("-"*30)
            # print([(x[0], x[1][1]) for x in determined])
            # print([(x[0], x[1][1]) for x in cands.top_n(joseki_num - len(determined))])

        solution = determined + cands.top_n(joseki_num - len(determined))  # TODO:ノードの共有があった場合には、親子関係が存在する可能性はある。解消するには全体を辿って検出する必要がありそう
        solution = [x[1] for x in solution]
        score = joseki_score(solution)
        print(len(solution), len(determined), len(cands.heap), score)

        if score > best_score:
            best_score = score
            best_solution = solution

        if score <= prev_score:
            count += 1
        else:
            count = 0

        # if count >= n_div_try:
        #     break

        prev_score = score

    total_visited_num = 0
    official_actions_list = []
    Glendenning_actions_list = []
    visit_num_list = []
    visit_rate_list = []
    win_rate_list = []
    for tree, action_list in best_solution:
        print(action_list)
        official_actions_list.append(" ".join(action_list))
        Glendenning_actions_list.append(",".join(list(map(Glendenning2Official, action_list))))
        visit_num_list.append(tree.visited_num)
        visit_rate_list.append(tree.visited_num / tree.game_num)
        win_rate_list.append(tree.p1_win_num / tree.visited_num)
        print("{} ({:.2f}%), p1 win rate = {:.2f}%".format(tree.visited_num, tree.visited_num / tree.game_num * 100, tree.p1_win_num / tree.visited_num * 100))
        total_visited_num += tree.visited_num
    print("total = {} ({:.2f}%)".format(total_visited_num, 100 * total_visited_num / opening_tree.game_num))

    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%y%m%d")

    df = pd.DataFrame({"official_actions_list": official_actions_list, "visit_num_list": visit_num_list, "visit_rate_list": visit_rate_list, "win_rate_list": win_rate_list})
    df.to_csv(os.path.join(save_dir, f"joseki_info_{formatted_date}.csv"), sep=",")

    np.savetxt(os.path.join(save_dir, f"joseki_{formatted_date}.txt"), np.array(Glendenning_actions_list, dtype=str), fmt="%s")
    # with open(os.path.join(save_dir, f"joseki_{formatted_date}.txt"), "w") as fout:
    #     for x in Glendenning_actions_list:
    #         fout.write(x)

    print()
    print(f"save result in {save_dir}")
