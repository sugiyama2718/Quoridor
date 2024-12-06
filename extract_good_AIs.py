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
from util import get_epoch_dir_name, generate_opening_tree, save_tree_graph, compute_contributions
tf.get_logger().setLevel(logging.ERROR)

SIGMA = 1.5
RANDOM_EPSILON = 1e-5  # 同スコアのものにランダム性を加える目的
EPSILON = 1e-10
FIX_EPOCH_LIST = [60, 61, 63, 65, 75, 81, 91, 130, 175, 215, 285, 325, 465, 775]  # eliminationの対象から外す 既にGUIでtraining用AIとして使用しているものなどを指定
EPOCH_CYCLE = 100
MAX_DEPTH = 20
#EXTRACT_GOOD_AIS_TAU = 0.48

# AIのデフォルトパラメータを設定
default_C_puct = 2.0
default_tau = 0.48
default_p_tau = 1.0

# 241201AI選定の設定
# EVALUATE_GAME_NUM = 10000
# SEARCHNODES_FOR_EXTRACT = 500
# START_CAND_NUM = 16
# START_DIFF_NUM = 8
# END_CAND_NUM = 4
# DIFF_R = 0.5
# ELIMINATE_NUM = 5  # AI消去処理の最大回数
# RATE_DIV = 0.5
# ELIMINATE_STEP = 1500  # 何試合ごとにAI消去処理を実施するか
# MAX_GAME_NUM_PER_EPOCH = 50

# elimination無し
EVALUATE_GAME_NUM = 10000
SEARCHNODES_FOR_EXTRACT = 500
START_CAND_NUM = 100
START_DIFF_NUM = 0
END_CAND_NUM = 100
DIFF_R = 0.5
ELIMINATE_NUM = 0
RATE_DIV = 0.5
ELIMINATE_STEP = 1000
MAX_GAME_NUM_PER_EPOCH = 50


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
    arg_i, arg_j, ai_param_i, ai_param_j, seed, wait_time = arg_tuple

    import time
    time.sleep(wait_time)

    from main import evaluate
    from CNNAI import CNNAI

    # AI1の作成
    if ai_param_i['AI_id'] == -1:
        AI1 = CNNAI(
            0,
            search_nodes=ai_param_i['search_nodes'],
            all_parameter_zero=True,
            p_is_almost_flat=True,
            seed=seed,
            C_puct=ai_param_i['C_puct'],
            tau=ai_param_i['tau'],
            p_tau=ai_param_i['p_tau']
        )
    else:
        AI1 = CNNAI(
            0,
            search_nodes=ai_param_i['search_nodes'],
            seed=seed,
            C_puct=ai_param_i['C_puct'],
            tau=ai_param_i['tau'],
            p_tau=ai_param_i['p_tau']
        )
        AI1.load(os.path.join(PARAMETER_DIR, get_epoch_dir_name(ai_param_i['AI_id']), "epoch{}.ckpt".format(ai_param_i['AI_id'])))

    # AI2の作成
    if ai_param_j['AI_id'] == -1:
        AI2 = CNNAI(
            1,
            search_nodes=ai_param_j['search_nodes'],
            all_parameter_zero=True,
            p_is_almost_flat=True,
            seed=seed,
            C_puct=ai_param_j['C_puct'],
            tau=ai_param_j['tau'],
            p_tau=ai_param_j['p_tau']
        )
    else:
        AI2 = CNNAI(
            1,
            search_nodes=ai_param_j['search_nodes'],
            seed=seed,
            C_puct=ai_param_j['C_puct'],
            tau=ai_param_j['tau'],
            p_tau=ai_param_j['p_tau']
        )
        AI2.load(os.path.join(PARAMETER_DIR, get_epoch_dir_name(ai_param_j['AI_id']), "epoch{}.ckpt".format(ai_param_j['AI_id'])))

    AIs = [AI1, AI2]
    AIs[0].tau = EVALUATION_TAU
    AIs[1].tau = EVALUATION_TAU

    ret = evaluate(AIs, 2, multiprocess=True, return_detail=True)
    del AIs
    return ret, arg_i, arg_j


def get_winner_from_action_list(action_list):
    # 勝者が先手なら1、後手なら-1、引き分けなら0を返す
    return -1 if len(action_list) % 2 == 0 else 1


def list_all_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        file_list.extend(files)
    return file_list

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--use_past_result', action='store_true', help='Use past result if specified')
    parser.add_argument('--use_past_id_only', action='store_true', help='Use past result if specified')
    args = parser.parse_args()

    use_past_result = args.use_past_result or args.use_past_id_only
    use_past_id_only = args.use_past_id_only
    load_array = use_past_result and not use_past_id_only

    param_files = list_all_files(PARAMETER_DIR)
    param_files = list(set([s.split(".")[0] for s in param_files]))
    param_files = [int(s[5:]) for s in param_files if s.startswith("epoch")]

    if use_past_result:
        rate_df = pd.read_csv(os.path.join(TRAIN_LOG_DIR, "detail", "estimated_rate.csv"))
        past_AI_id_arr = rate_df["AI_id"].values
        past_search_nodes_arr = rate_df["search nodes"].values

        # SEARCHNODES_FOR_EXTRACTのAIのみ使用
        is_normal_search_nodes_list = [x == SEARCHNODES_FOR_EXTRACT for x in past_search_nodes_arr]
        use_index_list = [i for i in range(len(past_AI_id_arr)) if is_normal_search_nodes_list[i]]
        past_AI_id_arr = past_AI_id_arr[use_index_list]
        past_search_nodes_arr = past_search_nodes_arr[use_index_list]

        all_AI_id_set = set([-1] + list(sorted(param_files)) + FIX_EPOCH_LIST)
        AI_id_set = all_AI_id_set - set(range(-1, max(past_AI_id_arr) + 1))
        AI_id_list = sorted(list(AI_id_set))

        if use_past_id_only:
            additional_AI_id_list = []
        else:
            ends_num = 0
            if ends_num > 0:
                additional_AI_id_list = [AI_id for AI_id in AI_id_list[:-ends_num] if AI_id % EPOCH_CYCLE == 0] + AI_id_list[-ends_num:]
            else:
                additional_AI_id_list = [AI_id for AI_id in AI_id_list if AI_id % EPOCH_CYCLE == 0]

        AI_id_list = list(past_AI_id_arr) + additional_AI_id_list
        search_nodes_list = list(past_search_nodes_arr) + [SEARCHNODES_FOR_EXTRACT] * len(additional_AI_id_list)

    else:
        AI_id_list = [-1] + list(sorted(param_files))
        # 選別(例)
        selection_criteria = [
            (0, 29, 1),
            (30, 199, 5),
            (200, 499, 20),
            (500, len(AI_id_list)-1, 100) 
        ]

        selected_AI_id_list = []
        for start_idx, end_idx, modulus in selection_criteria:
            end_idx = min(end_idx, len(AI_id_list)-1)
            selected_AI_id_list.extend([
                AI_id for AI_id in AI_id_list[start_idx:end_idx+1]
                if AI_id % modulus == 0
            ])

        AI_id_list = selected_AI_id_list
        search_nodes_list = [SEARCHNODES_FOR_EXTRACT] * len(AI_id_list)

    # ai_parametersを作成する
    ai_parameters = []

    # 元々あったパラメータ設定
    config_counter = 0
    for AI_id in AI_id_list:
        ai_param = {
            'config_id': config_counter,  # 内部識別用ID
            'AI_id': AI_id,
            'search_nodes': SEARCHNODES_FOR_EXTRACT,
            'C_puct': default_C_puct,
            'tau': default_tau,
            'p_tau': default_p_tau
        }
        ai_parameters.append(ai_param)
        config_counter += 1

    # 実験: tauを0.32に変更したバージョンも追加
    for AI_id in AI_id_list:
        ai_param = {
            'config_id': config_counter,  # 内部識別用ID
            'AI_id': AI_id,
            'search_nodes': SEARCHNODES_FOR_EXTRACT,
            'C_puct': default_C_puct,
            'tau': 0.32,
            'p_tau': default_p_tau
        }
        ai_parameters.append(ai_param)
        config_counter += 1

    # ai_parametersを元にAI_numを決定
    AI_num = len(ai_parameters)
    print(f"AI num = {AI_num}")

    sente_win_num_dict = {i: 0 for i in range(AI_num)}
    gote_win_num_dict = {i: 0 for i in range(AI_num)}
    action_lists_sente_dict = {i: [] for i in range(AI_num)}
    action_lists_gote_dict = {i: [] for i in range(AI_num)}

    dummy_AI_num = AI_num
    dummy_r_arr = np.log(np.arange(dummy_AI_num) + 1) + np.random.rand(dummy_AI_num)
    dummy_r_arr -= np.min(dummy_r_arr)

    n_arr = np.zeros((AI_num, AI_num))
    N_arr = np.zeros((AI_num, AI_num))

    if load_array and os.path.exists(os.path.join(TRAIN_LOG_DIR, "detail", "n_arr_extracted.csv")):
        past_n_arr = np.loadtxt(os.path.join(TRAIN_LOG_DIR, "detail", "n_arr_extracted.csv"), delimiter=',')
        past_N_arr = np.loadtxt(os.path.join(TRAIN_LOG_DIR, "detail", "N_arr_extracted.csv"), delimiter=',')
        past_n_arr = past_n_arr[np.ix_(use_index_list, use_index_list)]
        past_N_arr = past_N_arr[np.ix_(use_index_list, use_index_list)]
        size = past_n_arr.shape[0]
        n_arr[:size, :size] = past_n_arr
        N_arr[:size, :size] = past_N_arr

    shuffled_AI_index_list = list(range(AI_num))
    random.shuffle(shuffled_AI_index_list)
    matches = []
    for i in range(len(shuffled_AI_index_list) // 2):
        matches.append((shuffled_AI_index_list[i * 2], shuffled_AI_index_list[i * 2 + 1]))

    # 対戦していないAIが存在すると後ろの処理が正常に動作しないため、余りがいるときは別途matchを追加する
    if len(shuffled_AI_index_list) % 2 == 1:
        matches.append((shuffled_AI_index_list[-1], shuffled_AI_index_list[0]))

    def apply_matches(matches, total_game_num):
        args = []
        wait_time_list = [0] * len(matches)
        for i in range(len(matches)):
            wait_time_list[i] = i
        for k, (i, j) in enumerate(matches):
            # ここでai_parameters[i], ai_parameters[j]を渡す
            args.append((i, j, ai_parameters[i], ai_parameters[j], (k + total_game_num) * 10000, wait_time_list[k]))

        with Pool(processes=PROCESS_NUM) as p:
            imap = p.imap(func=evaluate_2game_process_2id, iterable=args)
            ret = list(tqdm(imap, total=len(matches)))

        for evaluate_ret, i, j in ret:
            sente_win_num, gote_win_num, draw_num, action_lists = evaluate_ret
            win_num_total = sente_win_num + gote_win_num
            n_arr[i, j] += win_num_total
            N_arr[i, j] += 2
            n_arr[j, i] += 2 - win_num_total
            N_arr[j, i] += 2

            for game_index, action_list in enumerate(action_lists):
                winner = get_winner_from_action_list(action_list)
                if game_index == 0:
                    if winner == 1:
                        sente_win_num_dict[i] += 1
                    elif winner == -1:
                        gote_win_num_dict[j] += 1
                    action_lists_sente_dict[i].append(action_list)
                    action_lists_gote_dict[j].append(action_list)
                elif game_index == 1:
                    if winner == 1:
                        sente_win_num_dict[j] += 1
                    elif winner == -1:
                        gote_win_num_dict[i] += 1
                    action_lists_sente_dict[j].append(action_list)
                    action_lists_gote_dict[i].append(action_list)

    cand_num = START_CAND_NUM
    diff_num = START_DIFF_NUM

    save_dir = os.path.join(TRAIN_LOG_DIR, "detail")
    os.makedirs(save_dir, exist_ok=True)

    kifu_save_dir = os.path.join(save_dir, "kifu")
    os.makedirs(kifu_save_dir, exist_ok=True)

    all_eliminate_AI_indices_list = []
    survived_list = list(range(AI_num))
    total_game_num = 0
    eliminate_count = 0
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

        diversity_sente_dict = {}
        diversity_gote_dict = {}

        for i in survived_list:
            # AI_id取得にはai_parameters[i]['AI_id']を使用
            AI_id = ai_parameters[i]['AI_id']
            kifu_sente = action_lists_sente_dict[i]
            kifu_gote = action_lists_gote_dict[i]

            kifu_tree_sente, statevec2node_sente = generate_opening_tree(kifu_sente, MAX_DEPTH, disable_tqdm=True)
            (_, diversity_sente), (_, _) = compute_contributions(kifu_tree_sente, statevec2node_sente, len(kifu_sente), MAX_DEPTH, is_print=False)
            diversity_sente_dict[i] = diversity_sente

            kifu_tree_gote, statevec2node_gote = generate_opening_tree(kifu_gote, MAX_DEPTH, disable_tqdm=True)
            (_, _), (_, diversity_gote) = compute_contributions(kifu_tree_gote, statevec2node_gote, len(kifu_gote), MAX_DEPTH, is_print=False)
            diversity_gote_dict[i] = diversity_gote

        survived_ai_params = [ai_parameters[i] for i in survived_list]
        sente_win_nums = [sente_win_num_dict[i] for i in survived_list]
        gote_win_nums = [gote_win_num_dict[i] for i in survived_list]
        diversity_sente_list = [diversity_sente_dict.get(i, 0) for i in survived_list]
        diversity_gote_list = [diversity_gote_dict.get(i, 0) for i in survived_list]

        r_df = pd.DataFrame({
            "AI_id": [param['AI_id'] for param in survived_ai_params],
            "search_nodes": [param['search_nodes'] for param in survived_ai_params],
            "C_puct": [param['C_puct'] for param in survived_ai_params],
            "tau": [param['tau'] for param in survived_ai_params],
            "p_tau": [param['p_tau'] for param in survived_ai_params],
            "rate": estimated_r_arr[survived_list],
            "sente_win_num": sente_win_nums,
            "gote_win_num": gote_win_nums,
            "diversity_sente": diversity_sente_list,
            "diversity_gote": diversity_gote_list,
        })

        r_df.to_csv(os.path.join(save_dir, "estimated_rate.csv"), index=False)

        np.savetxt(os.path.join(save_dir, "n_arr.csv"), n_arr, delimiter=",")
        np.savetxt(os.path.join(save_dir, "N_arr.csv"), N_arr, delimiter=",")

        np.savetxt(os.path.join(save_dir, "n_arr_extracted.csv"), n_arr[np.ix_(survived_list, survived_list)], delimiter=",")
        np.savetxt(os.path.join(save_dir, "N_arr_extracted.csv"), N_arr[np.ix_(survived_list, survived_list)], delimiter=",")

        plt.clf()
        plt.plot([param['AI_id'] for param in survived_ai_params], estimated_r_arr[survived_list], label="est")
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
                if increment_dict[cls] > leave_num and ai_parameters[AI_index]['AI_id'] not in FIX_EPOCH_LIST:
                    eliminate_AI_indices.append(AI_index)
            all_eliminate_AI_indices_list = sorted(list(set(all_eliminate_AI_indices_list) | set(eliminate_AI_indices)))
            survived_list = sorted(list(set(range(AI_num)) - set(all_eliminate_AI_indices_list)))
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

    # 棋譜保存処理
    for i in survived_list:
        AI_id = ai_parameters[i]['AI_id']

        kifu_sente = action_lists_sente_dict[i]
        kifu_gote = action_lists_gote_dict[i]

        # 再度ツリー生成(必要であればキャッシュを使うなど工夫可能)
        kifu_tree_sente, statevec2node_sente = generate_opening_tree(kifu_sente, MAX_DEPTH, disable_tqdm=True)
        tree_file_sente = os.path.join(kifu_save_dir, f"AI_{AI_id}_sente_tree")
        save_tree_graph(kifu_tree_sente, statevec2node_sente, tree_file_sente)

        sente_file_path = os.path.join(kifu_save_dir, f"AI_{AI_id}_sente.txt")
        with open(sente_file_path, 'w') as f:
            for kifu in kifu_sente:
                f.write(','.join(kifu) + '\n')

        kifu_tree_gote, statevec2node_gote = generate_opening_tree(kifu_gote, MAX_DEPTH, disable_tqdm=True)
        tree_file_gote = os.path.join(kifu_save_dir, f"AI_{AI_id}_gote_tree")
        save_tree_graph(kifu_tree_gote, statevec2node_gote, tree_file_gote)

        gote_file_path = os.path.join(kifu_save_dir, f"AI_{AI_id}_gote.txt")
        with open(gote_file_path, 'w') as f:
            for kifu in kifu_gote:
                f.write(','.join(kifu) + '\n')
