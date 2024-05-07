import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from config import *
from calc_multi_rate import estimate_multi_rate
from collections import Counter
import copy
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

SIGMA = 1.0
RANDOM_EPSILON = 1e-5  # 同スコアのものにランダム性を加える目的
EPSILON = 1e-10
FIX_EPOCH_LIST = [60, 62, 71, 91, 96, 155, 220, 465, 620, 1050, 1284, 2020, 2520, 2780, 2910, 3090, 3350, 3360, 3400, 3460, 3545, 3560, 3700, 3850, 3985]  # eliminationの対象から外す 既にGUIでtraining用AIとして使用しているものなどを指定

USE_PAST_RESULT = True

# EVALUATE_GAME_NUM = 14000
# SEARCHNODES_FOR_EXTRACT = 500
# START_CAND_NUM = 50
# START_DIFF_NUM = 100
# END_CAND_NUM = 5
# DIFF_R = 0.5
# ELIMINATE_NUM = 8
# RATE_DIV = 0.5
# ELIMINATE_STEP = 1000

# 240106AI選定の設定
# EVALUATE_GAME_NUM = 10000
# SEARCHNODES_FOR_EXTRACT = 500
# START_CAND_NUM = 32
# START_DIFF_NUM = 16
# END_CAND_NUM = 4
# DIFF_R = 0.5
# ELIMINATE_NUM = 6  # AI消去処理の最大回数
# RATE_DIV = 0.5
# ELIMINATE_STEP = 1500  # 何試合ごとにAI消去処理を実施するか

# 240126AI選定の設定
EVALUATE_GAME_NUM = 7000
SEARCHNODES_FOR_EXTRACT = 500
START_CAND_NUM = 32
START_DIFF_NUM = 16
END_CAND_NUM = 4
DIFF_R = 0.5
ELIMINATE_NUM = 6  # AI消去処理の最大回数
RATE_DIV = 0.5
ELIMINATE_STEP = 1000  # 何試合ごとにAI消去処理を実施するか
MAX_GAME_NUM_PER_EPOCH = 50

# elimination無し
# EVALUATE_GAME_NUM = 10000
# SEARCHNODES_FOR_EXTRACT = 500
# START_CAND_NUM = 100
# START_DIFF_NUM = 0
# END_CAND_NUM = 100
# DIFF_R = 0.5
# ELIMINATE_NUM = 0
# RATE_DIV = 0.5
# ELIMINATE_STEP = 1000

# EVALUATE_GAME_NUM = 50
# START_CAND_NUM = 3
# START_DIFF_NUM = 2
# END_CAND_NUM = 1
# DIFF_R = 0.5
# ELIMINATE_NUM = 2
# RATE_DIV = 0.5


def calc_match_score_arr(N_arr, r_arr):
    # 各AI一回は対戦しているとする
    r_mat1 = np.broadcast_to(r_arr, (AI_num, AI_num))
    r_mat2 = r_mat1.T
    kernel = np.exp(-np.square(r_mat1 - r_mat2) / (2 * SIGMA ** 2))

    # 自分自身は省く
    for i in range(kernel.shape[0]):
        kernel[i, i] = 0
    kernel = kernel / np.sum(kernel, axis=1).reshape((-1, 1))
    #kernel = np.triu(kernel, k=1)
    
    N_sum_arr = np.sum(N_arr, axis=1)
    N_max = np.max(N_sum_arr)

    # kernel...レートの近さ。近いほど高スコア  -N\arr/N_sum_arr...マッチング回数。少ないほど高スコア -N_sum_arr/N_max...そのAIの総試合回数。少ないほど高スコア
    ret = np.maximum(2 + kernel - N_arr / N_sum_arr - N_sum_arr / N_max, np.zeros_like(kernel)) + RANDOM_EPSILON * np.random.rand(AI_num, AI_num)
    ret = np.triu(ret, k=1)
    return ret


def evaluate_2game_process_2id(arg_tuple):
    from main import evaluate
    from CNNAI import CNNAI
    # 先後で２試合して勝利数を返す
    arg_i, arg_j, AI_id1, AI_id2, seed, search_nodes1, search_nodes2 = arg_tuple

    if AI_id1 == -1:
        AI1 = CNNAI(0, search_nodes=search_nodes1, all_parameter_zero=True, p_is_almost_flat=True, seed=seed)
    else:
        AI1 = CNNAI(0, search_nodes=search_nodes1, seed=seed)
        AI1.load(os.path.join(PARAMETER_DIR, "epoch{}.ckpt".format(AI_id1)))

    if AI_id2 == -1:
        AI2 = CNNAI(1, search_nodes=search_nodes2, all_parameter_zero=True, p_is_almost_flat=True, seed=seed)
    else:
        AI2 = CNNAI(1, search_nodes=search_nodes2, seed=seed)
        AI2.load(os.path.join(PARAMETER_DIR, "epoch{}.ckpt".format(AI_id2)))

    AIs = [AI1, AI2]
    AIs[0].tau = EVALUATION_TAU
    AIs[1].tau = EVALUATION_TAU

    ret = evaluate(AIs, 2, multiprocess=True)
    del AIs
    return ret, arg_i, arg_j

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    param_files = os.listdir(PARAMETER_DIR)
    param_files = list(set([s.split(".")[0] for s in param_files]))
    param_files = [int(s[5:]) for s in param_files if s.startswith("epoch")]
    
    if USE_PAST_RESULT:
        rate_df = pd.read_csv(os.path.join(TRAIN_LOG_DIR, "detail", "estimated_rate.csv"))
        past_AI_id_arr = rate_df["AI_id"].values
        past_search_nodes_arr = rate_df["search nodes"].values

        # SEARCHNODES_FOR_EXTRACTのAIのみを読み込む。他の探索数のも読み込みたい場合コメントアウト
        is_normal_search_nodes_list = [x == SEARCHNODES_FOR_EXTRACT for x in past_search_nodes_arr]
        use_index_list = [i for i in range(len(past_AI_id_arr)) if is_normal_search_nodes_list[i]]
        past_AI_id_arr = past_AI_id_arr[use_index_list]
        past_search_nodes_arr = past_search_nodes_arr[use_index_list]

        # print(past_AI_id_list)
        # print(past_search_nodes_list)
        # print(len(past_AI_id_list), len(past_search_nodes_list))

        # ノード数固定で良いパラメータ抜き出し用
        all_AI_id_set = set([-1] + list(sorted(param_files)) + FIX_EPOCH_LIST)
        AI_id_set = all_AI_id_set - set(range(-1, max(past_AI_id_arr) + 1))
        AI_id_list = sorted(list(AI_id_set))

        #ends_num = 100
        ends_num = 0
        if ends_num > 0:
            additional_AI_id_list = [AI_id for AI_id in AI_id_list[:-ends_num] if AI_id % 5 == 0] + AI_id_list[-ends_num:]
        else:
            additional_AI_id_list = [AI_id for AI_id in AI_id_list if AI_id % 5 == 0]
        AI_id_list = list(past_AI_id_arr) + additional_AI_id_list
        search_nodes_list = list(past_search_nodes_arr) + [SEARCHNODES_FOR_EXTRACT] * len(additional_AI_id_list)
        # print(AI_id_list)
        # print(search_nodes_list)
        # print(len(AI_id_list), len(search_nodes_list))
        # exit()

        # ノード数複数評価版
        # AI_id_list = past_AI_id_list
        # original_AI_id_list = copy.copy(AI_id_list)
        # strong_AI_list = [x for x in AI_id_list if x >= 2000]
        # AI_id_list = AI_id_list + strong_AI_list * 3
        # search_nodes_list = [500] * len(original_AI_id_list) + [200] * len(strong_AI_list) + [300] * len(strong_AI_list) + [1000] * len(strong_AI_list)
        # SEARCH1_TARGET_LIST = [2910]
        # AI_id_list += SEARCH1_TARGET_LIST  # policy networkのみの強さを測定する
        # search_nodes_list += len(SEARCH1_TARGET_LIST) * [1]
        
        print(list(zip(AI_id_list, search_nodes_list)))
    else:
        AI_id_list = [-1] + list(sorted(param_files))
        both_ends_num = 100  # AI_id_listの両端をいくつそのまま残すか
        AI_id_list = AI_id_list[:both_ends_num] + [AI_id for AI_id in AI_id_list[both_ends_num:-both_ends_num] if AI_id % 5 == 0] + AI_id_list[-both_ends_num:]
        search_nodes_list = [EVALUATION_SEARCHNODES] * len(AI_id_list)
        print(AI_id_list)

    AI_num = len(AI_id_list)
    print(f"AI num = {AI_num}")

    dummy_AI_num = AI_num
    dummy_r_arr = np.log(np.arange(dummy_AI_num) + 1) + np.random.rand(dummy_AI_num)
    dummy_r_arr -= np.min(dummy_r_arr)

    def dummy_evaluate_2game_process(arg_tuple):
        arg_i, arg_j, AI_id1, AI_id2, seed, search_nodes1, search_nodes2 = arg_tuple
        p = 1 / (1 + np.exp(dummy_r_arr[arg_j] - dummy_r_arr[arg_i]))
        ret = []
        for _ in range(2):
            ret.append(int(random.random() < p))
        return sum(ret), arg_i, arg_j

    n_arr = np.zeros((AI_num, AI_num))
    N_arr = np.zeros((AI_num, AI_num))

    # train_logにn_arrが残っている場合、AI_idとsearch_nodesがそのときのものと一致している前提で新しいn_arr, N_arrに読み込む
    if os.path.exists(os.path.join(TRAIN_LOG_DIR, "detail", "n_arr_extracted.csv")):
        past_n_arr = np.loadtxt(os.path.join(TRAIN_LOG_DIR, "detail", "n_arr_extracted.csv"), delimiter=',')
        past_N_arr = np.loadtxt(os.path.join(TRAIN_LOG_DIR, "detail", "N_arr_extracted.csv"), delimiter=',')
        past_n_arr = past_n_arr[np.ix_(use_index_list, use_index_list)]
        past_N_arr = past_N_arr[np.ix_(use_index_list, use_index_list)]
        size = past_n_arr.shape[0]
        n_arr[:size, :size] = past_n_arr
        N_arr[:size, :size] = past_N_arr
        # print(past_n_arr)
        # print(past_N_arr)
        # print(past_N_arr.shape)
        # print(np.sum(past_N_arr, axis=1))
        # print(np.sum(past_n_arr, axis=1) / np.sum(past_N_arr, axis=1))
        # exit()

    shuffled_AI_index_list = list(range(len(AI_id_list)))
    random.shuffle(shuffled_AI_index_list)
    matches = []
    for i in range(len(shuffled_AI_index_list) // 2):
        matches.append((shuffled_AI_index_list[i * 2], shuffled_AI_index_list[i * 2 + 1]))

    # 対戦していないAIが存在すると後ろの処理が正常に動作しないため、余りがいるときは別途matchを追加する
    if len(shuffled_AI_index_list) % 2 == 1:
        matches.append((shuffled_AI_index_list[-1], shuffled_AI_index_list[0]))
        # print("add match")
        # print(matches)

    def apply_matches(matches, total_game_num):
        args = []
        for k, (i, j) in enumerate(matches):
            args.append((i, j, AI_id_list[i], AI_id_list[j], (k + total_game_num) * 10000, search_nodes_list[i], search_nodes_list[j]))

        with Pool(processes=PROCESS_NUM) as p:
            imap = p.imap(func=evaluate_2game_process_2id, iterable=args)
            #imap = p.imap(func=dummy_evaluate_2game_process, iterable=args)
            ret = list(tqdm(imap, total=len(matches)))

        for game_result, i, j in ret:
            n_arr[i, j] += game_result
            N_arr[i, j] += 2
            n_arr[j, i] += 2 - game_result
            N_arr[j, i] += 2

    cand_num = START_CAND_NUM
    diff_num = START_DIFF_NUM

    save_dir = os.path.join(TRAIN_LOG_DIR, "detail")
    os.makedirs(save_dir, exist_ok=True)

    all_eliminate_AI_indices_list = []
    survived_list = range(len(AI_id_list))
    total_game_num = 0
    eliminate_count = 0
    #estimated_r_arr = np.zeros(AI_num)
    epoch = 0
    while total_game_num < EVALUATE_GAME_NUM:
        epoch += 1
        total_game_num += len(matches) * 2
        print("="*50)
        print(epoch)
        print("game num = {}".format(len(matches) * 2))
        print("total game num = {}".format(total_game_num))

        apply_matches(matches, total_game_num)

        def estimate_multi_rate_process(args):
            n_arr, N_arr, AI_num = args
            return estimate_multi_rate(n_arr, N_arr, AI_num)
        
        with Pool(processes=1) as p:
            imap = p.imap(func=estimate_multi_rate_process, iterable=[(n_arr, N_arr, AI_num)])
            estimated_r_arr = list(imap)[0]

        r_df = pd.DataFrame({"AI_id": np.array(AI_id_list)[survived_list], "search nodes": np.array(search_nodes_list)[survived_list], "rate": estimated_r_arr[survived_list]})
        r_df.to_csv(os.path.join(save_dir, "estimated_rate.csv"))

        np.savetxt(os.path.join(save_dir, "n_arr.csv"), n_arr, delimiter=",")
        np.savetxt(os.path.join(save_dir, "N_arr.csv"), N_arr, delimiter=",")

        np.savetxt(os.path.join(save_dir, "n_arr_extracted.csv"), n_arr[np.ix_(survived_list, survived_list)], delimiter=",")
        np.savetxt(os.path.join(save_dir, "N_arr_extracted.csv"), N_arr[np.ix_(survived_list, survived_list)], delimiter=",")

        plt.clf()
        #plt.plot(np.array(AI_id_list)[survived_list], r_arr[survived_list], label="true")
        plt.plot(np.array(AI_id_list)[survived_list], estimated_r_arr[survived_list], label="est")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "estimated_rate_graph.png"))

        if total_game_num // ELIMINATE_STEP > eliminate_count and eliminate_count < ELIMINATE_NUM:
            rate_class = np.floor(estimated_r_arr / RATE_DIV).astype(int)
            class_counter = Counter(rate_class)

            print("--- elimination ---")
            print("cand num = {}, diff num = {}".format(cand_num, diff_num))

            increment_dict = {}
            for cls in class_counter.keys():
                increment_dict[cls] = 0
                print("{}: {}".format(cls, cand_num + cls * diff_num))

            eliminate_AI_indices = []
            for AI_index, cls in enumerate(rate_class):
                increment_dict[cls] += 1
                leave_num = cand_num + cls * diff_num
                if increment_dict[cls] > leave_num and AI_id_list[AI_index] not in FIX_EPOCH_LIST:
                    eliminate_AI_indices.append(AI_index)
            all_eliminate_AI_indices_list = sorted(list(set(all_eliminate_AI_indices_list) | set(eliminate_AI_indices)))
            survived_list = sorted(list(set(range(len(AI_id_list))) - set(all_eliminate_AI_indices_list)))
            print("eliminated num = {}".format(len(all_eliminate_AI_indices_list)))
            print("survived num = {}".format(len(survived_list)))

            eliminate_count += 1
            progress = eliminate_count / (ELIMINATE_NUM - 1)
            cand_num = int(progress * END_CAND_NUM + (1 - progress) * START_CAND_NUM)
            diff_num = int(diff_num * DIFF_R)
            

        N_sum_arr = np.sum(N_arr, axis=1)

        possible_games = 2 * np.ones_like(N_sum_arr)
        possible_games[all_eliminate_AI_indices_list] = 0
        
        match_score_arr = calc_match_score_arr(N_arr, estimated_r_arr)

        matches = []
        while np.sum(possible_games) >= 10 and np.max(match_score_arr) > EPSILON and 2 * len(matches) < MAX_GAME_NUM_PER_EPOCH:
            match_index = np.argmax(match_score_arr)
            i = match_index // AI_num
            j = match_index % AI_num
            possible_games[i] -= 2
            possible_games[j] -= 2
            match_score_arr[i, j] = 0

            if possible_games[i] < EPSILON:
                match_score_arr[i, :] = 0
                match_score_arr[:, i] = 0
            if possible_games[j] < EPSILON:
                match_score_arr[j, :] = 0
                match_score_arr[:, j] = 0
            
            matches.append((i, j))


