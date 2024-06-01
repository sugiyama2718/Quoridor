# coding:utf-8
#from memory_profiler import profile
from Agent import actionid2str
from State import State, CHANNEL, State_init, eq_state, accept_action_str, BOARD_LEN, get_player_dist_from_goal, calc_dist_array, display_cui, feature_CNN, get_row_wall, get_column_wall
from Human import Human
from CNNAI import CNNAI
from BasicAI import state_copy
import time
import sys
import numpy as np
import gc
import h5py
import os
import datetime
from subprocess import check_output
from rating import calc_rate
import pickle
from multiprocessing import Pool
from copy import deepcopy
from config import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import csv
from collections import OrderedDict
from analyze_h5 import analyze_h5_main
import shutil
import random

# agentsは初期化されてるとする
def normal_play(agents, initial_state=None):
    if initial_state is None:
        state = State()
        State_init(state)
    else:
        state = initial_state
        
    while True:
        display_cui(state)
        start = time.time()
        s = agents[0].act(state, showNQ=True)
        end = time.time()
        if isinstance(s, int):
            a = actionid2str(state, s)
        else:
            a = s
        while not accept_action_str(state, a):
            print(a)
            print("this action is impossible")
            s = agents[0].act(state, showNQ=True)
            if isinstance(s, int):
                a = actionid2str(state, s)
            else:
                a = s
        agents[1].prev_action = s

        # g = agents[0].get_tree_for_graphviz()
        # g.render(os.path.join("game_trees", "game_tree{}".format(state.turn)))

        if state.terminate:
            break
        #time.sleep(0.1)

        display_cui(state)
        s = agents[1].act(state, showNQ=True)
        if isinstance(s, int):
            a = actionid2str(state, s)
        else:
            a = s
        while not accept_action_str(state, a):
            print(a)
            print("this action is impossible")
            s = agents[1].act(state, showNQ=True)
            if isinstance(s, int):
                a = actionid2str(state, s)
            else:
                a = s
        agents[0].prev_action = s

        # g = agents[1].get_tree_for_graphviz()
        # g.render(os.path.join("game_trees", "game_tree{}".format(state.turn)))

        #time.sleep(0.1)
        if state.terminate:
            break

    display_cui(state)
    print("The game finished. reward=({}, {})".format(state.reward, -state.reward))
    return state.reward


def generate_data(AIs, play_num, noise=NOISE, display=False, equal_draw=False, info=False, id_=None):
    data = []
    hash_ = 0
    for j in range(play_num):
        state = State()
        State_init(state)
        AIs[0].init_prev()
        AIs[1].init_prev()
        featuress = [[], [], [], []]
        for i, b1, b2 in [(0, False, False), (1, True, False), (2, False, True), (3, True, True)]:
            featuress[i].append(feature_CNN(state, b1, b2))

        pis = []
        states = [state_copy(state)]
        v_prevs = []
        v_posts = []
        searched_node_nums = []
        move_count_list = []  # B, Wの移動数をターンごと格納
        move_count = [0, 0]
        B_xy_list = []
        W_xy_list = []
        tau = np.random.rand() * (TAU_MAX - TAU_MIN_OPENING) + TAU_MIN_OPENING
        AIs[0].tau = tau
        AIs[1].tau = tau
        while True:
            if state.turn >= 20:
                AIs[0].tau = TAU_MIN
                AIs[1].tau = TAU_MIN
            s, pi, v_prev, v_post, searched_node_num = AIs[0].act_and_get_pi(state, noise=noise, showNQ=display, opponent_prev_tree=AIs[1].prev_tree)
            a = actionid2str(state, s)
            while not accept_action_str(state, a):
                print("this action is impossible")
                print(a)
                display_cui(state)
                exit()
            AIs[1].prev_action = s

            pis.append(pi)
            v_prevs.append(v_prev)
            v_posts.append(v_post)
            searched_node_nums.append(searched_node_num)
            move_count_list.append(tuple(move_count))
            if len(a) == 2:  # aが移動を表す文字列なら2文字になる
                move_count[1 - state.turn % 2] += 1
            B_xy_list.append((state.Bx, state.By))
            W_xy_list.append((state.Wx, state.Wy))

            if display:
                print("generate id=", id_)
                display_cui(state)
            end = False
            for state2 in states:
                if equal_draw and eq_state(state, state2):
                    end = True
                    break
            if end:
                break
            states.append(state_copy(state))
            if state.terminate:
                break
            for i, b1, b2 in [(0, False, False), (1, True, False), (2, False, True), (3, True, True)]:
                featuress[i].append(feature_CNN(state, b1, b2))
            s, pi, v_prev, v_post, searched_node_num = AIs[1].act_and_get_pi(state, noise=noise, showNQ=display, opponent_prev_tree=AIs[0].prev_tree)
            a = actionid2str(state, s)
            while not accept_action_str(state, a):
                print("this action is impossible")
                print(a)
                display_cui(state)
                exit()
            AIs[0].prev_action = s

            pis.append(pi)
            v_prevs.append(-v_prev)  # 後手基準のvが返るので、先手基準のvに直す
            v_posts.append(-v_post)
            searched_node_nums.append(searched_node_num)
            move_count_list.append(tuple(move_count))
            if len(a) == 2:  # aが移動を表す文字列なら2文字になる
                move_count[1 - state.turn % 2] += 1
            B_xy_list.append((state.Bx, state.By))
            W_xy_list.append((state.Wx, state.Wy))

            if display:
                print("generate id=", id_)
                display_cui(state)
            end = False
            for state2 in states:
                if equal_draw and eq_state(state, state2):
                    end = True
                    break
            if end:
                break
            states.append(state_copy(state))
            if state.terminate:
                break
            for i, b1, b2 in [(0, False, False), (1, True, False), (2, False, True), (3, True, True)]:
                featuress[i].append(feature_CNN(state, b1, b2))
        del states

        hash_ += state.turn
        if state.reward == 0:
            continue

        # stateは終端状態になっている
        B_dist, W_dist = get_player_dist_from_goal(state)
        dist_diff = W_dist - B_dist  # 何マス差で勝ったか。勝ちで正になるよう、W-Bにしている
        all_turn_num = state.turn
        move_count[0] += B_dist
        move_count[1] += W_dist

        dist_array1 = calc_dist_array(state, 0)
        dist_array2 = calc_dist_array(state, BOARD_LEN - 1)

        def calc_traversed_arr_list(xy_list):
            traversed_arr = np.zeros((9, 9))
            traversed_arr_list = []
            for x, y in xy_list[::-1]:
                traversed_arr[x, y] = 1.0
                traversed_arr_list.append(np.copy(traversed_arr))
            traversed_arr_list = traversed_arr_list[::-1]
            return traversed_arr_list
        
        B_traversed_arr_list = calc_traversed_arr_list(B_xy_list)
        W_traversed_arr_list = calc_traversed_arr_list(W_xy_list)

        next_pis = pis[1:] + [np.zeros((137,))]

        def pi_flip1(pi):
            a = np.flip(pi[:64].reshape((8, 8)), 0).flatten()
            b = np.flip(pi[64:128].reshape((8, 8)), 0).flatten()
            mvarray1 = pi[128:].reshape((3, 3))
            mvarray2 = np.zeros((3, 3))
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    mvarray2[x, y] = mvarray1[-x, y]
            c = mvarray2.flatten()
            return np.concatenate([a, b, c])
        
        def pi_flip2(pi):
            a = np.flip(pi[:64].reshape((8, 8)), 1).flatten()
            b = np.flip(pi[64:128].reshape((8, 8)), 1).flatten()
            mvarray1 = pi[128:].reshape((3, 3))
            mvarray2 = np.zeros((3, 3))
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    mvarray2[x, y] = mvarray1[x, -y]
            c = mvarray2.flatten()
            return np.concatenate([a, b, c])
        
        def pi_flip3(pi):
            a = np.flip(np.flip(pi[:64].reshape((8, 8)), 1), 0).flatten()
            b = np.flip(np.flip(pi[64:128].reshape((8, 8)), 1), 0).flatten()
            mvarray1 = pi[128:].reshape((3, 3))
            mvarray2 = np.zeros((3, 3))
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    mvarray2[x, y] = mvarray1[-x, -y]
            c = mvarray2.flatten()
            return np.concatenate([a, b, c])

        row_wall = get_row_wall(state)
        column_wall = get_column_wall(state)
        for turn, feature1, feature2, feature3, feature4, pi, v_prev, v_post, searched_node_num, mid_move_count, B_traversed_arr, W_traversed_arr, next_pi in zip(
            range(all_turn_num), featuress[0], featuress[1], featuress[2], featuress[3], 
            pis, v_prevs, v_posts, searched_node_nums, move_count_list, B_traversed_arr_list, W_traversed_arr_list, next_pis):

            data.append((feature1, pi, state.reward, v_prev, v_post, searched_node_num, 
                         dist_diff, state.black_walls, state.white_walls, all_turn_num - turn, move_count[0] - mid_move_count[0], move_count[1] - mid_move_count[1],
                         row_wall, column_wall, dist_array1, dist_array2, B_traversed_arr, W_traversed_arr, next_pi))

            data.append((feature2, pi_flip1(pi), state.reward, v_prev, v_post, searched_node_num, 
                         dist_diff, state.black_walls, state.white_walls, all_turn_num - turn, move_count[0] - mid_move_count[0], move_count[1] - mid_move_count[1],
                         np.flip(row_wall, 0), np.flip(column_wall, 0), np.flip(dist_array1, 0), np.flip(dist_array2, 0), np.flip(B_traversed_arr, 0), np.flip(W_traversed_arr, 0),
                         pi_flip1(next_pi)))

            data.append((feature3, pi_flip2(pi), -state.reward, -v_prev, -v_post, searched_node_num, 
                         -dist_diff, state.white_walls, state.black_walls, all_turn_num - turn, move_count[1] - mid_move_count[1], move_count[0] - mid_move_count[0],
                         np.flip(row_wall, 1), np.flip(column_wall, 1), np.flip(dist_array2, 1), np.flip(dist_array1, 1), np.flip(W_traversed_arr, 1), np.flip(B_traversed_arr, 1),
                         pi_flip2(next_pi)))

            data.append((feature4, pi_flip3(pi), -state.reward, -v_prev, -v_post, searched_node_num, 
                         -dist_diff, state.white_walls, state.black_walls, all_turn_num - turn, move_count[1] - mid_move_count[1], move_count[0] - mid_move_count[0],
                         np.flip(np.flip(row_wall, 1), 0), np.flip(np.flip(column_wall, 1), 0), np.flip(np.flip(dist_array2, 1), 0), np.flip(np.flip(dist_array1, 1), 0), np.flip(np.flip(W_traversed_arr, 1), 0), np.flip(np.flip(B_traversed_arr, 1), 0),
                         pi_flip3(next_pi)))
    if info:
        print("hash = {}".format(hash_))

    return data


# 中で先手後手を順番に入れ替えている
def evaluate(AIs, play_num, return_draw=False, multiprocess=False, display=False):
    wins = 0.
    draw_num = 0
    total_time_without_endgame = 0.0
    total_turn_without_endgame = 0
    for i in range(play_num):
        game_start_time = time.time()
        is_endgame = False
        state = State()
        State_init(state)
        AIs[0].init_prev()
        AIs[1].init_prev()
        AIs[i % 2].color = 0
        AIs[1 - i % 2].color = 1
        while True:
            if display:
                display_cui(state)
            s, pi, v_prev, v_post, _ = AIs[i % 2].act_and_get_pi(state)
            a = actionid2str(state, s)
            while not accept_action_str(state, a):
                print("this action is impossible")
                s, pi, v_prev, v_post, _ = AIs[i % 2].act_and_get_pi(state)
                a = actionid2str(state, s)
            AIs[1 - i % 2].prev_action = s

            if not is_endgame and state.pseudo_terminate:
                total_time_without_endgame += time.time() - game_start_time
                total_turn_without_endgame += state.turn
                is_endgame = True

            if state.terminate:
                break

            if display:
                display_cui(state)

            s, pi, v_prev, v_post, _ = AIs[1 - i % 2].act_and_get_pi(state)
            a = actionid2str(state, s)
            while not accept_action_str(state, a):
                print("this action is impossible")
                s, pi, v_prev, v_post, _ = AIs[1 - i % 2].act_and_get_pi(state)
                a = actionid2str(state, s)
            AIs[i % 2].prev_action = s

            if not is_endgame and state.pseudo_terminate:
                total_time_without_endgame += time.time() - game_start_time
                total_turn_without_endgame += state.turn
                is_endgame = True

            if state.terminate:
                break

        if i % 2 == 0 and state.reward == 1:
            wins += 1.
        elif i % 2 == 1 and state.reward == -1:
            wins += 1.
        elif state.reward == 0:
            wins += 0.5
            draw_num += 1
        if not multiprocess:
            sys.stderr.write('\r\033[K {}win/{}'.format(i + 1 - wins, i + 1))
            sys.stderr.flush()
    if not multiprocess:
        print("")
    
    AIs[0].color = 0
    AIs[1].color = 1

    if display:
        print("total_time_without_endgame = {:.3f}s, time per one move = {:.3f}s".format(total_time_without_endgame, total_time_without_endgame / total_turn_without_endgame))

    if return_draw:
        return wins, draw_num
    else:
        return wins


def debug_learn(AIs):
    AIs[1].learn(H5_NUM - 1, H5_NUM, LEARN_REP_NUM, TRAIN_ARRAY_SIZE)
    # ----------evaluation--------
    AIs[0].tau = TAU_MIN
    AIs[1].tau = TAU_MIN
    AIs[0].search_nodes = search_nodes
    AIs[1].search_nodes = search_nodes

    play_num = 100
    white_win_num = evaluate(AIs, play_num)
    win_rate = (play_num - white_win_num) / play_num
    print("new AI win rate={}".format(win_rate))
    AIs[1].save(os.path.join(PARAMETER_DIR, "debug.ckpt"))


def train_from_random_parameter_process(x):
    # filters = 32
    # layer_num = 9
    first_epoch = 2077
    last_epoch = 2257
    # first_epoch = 2125 + 60
    # last_epoch = 2365
    lr_schedule = [(last_epoch - first_epoch + 1, LEARNING_RATE)] 
    #lr_schedule = [(10, LEARNING_RATE * 10), (100, LEARNING_RATE * 3), (last_epoch - first_epoch + 1, LEARNING_RATE)]  # (lrを変える最初から数えたepoch数, lr)のリスト。epoch数で昇順。

    s = time.time()
    
    #repeat = LEARN_REP_NUM
    ckpt_path = "train_results"
    data_path = "data_for_experiment/230928/data"
    #data_path = "train_results/data"

    AI = CNNAI(0)
    AI.save(os.path.join(ckpt_path, "parameter/train_experiment.ckpt"))

    loss_dict = {"train_loss": 5.0, "v_loss": 1.0, "v_reg_loss": 1.0, "p_loss": 5.0, "p_reg_loss": 5.0}
    valid_loss = 5.0
    for name in AUX_NAME_LIST:
        loss_dict[name] = 5.0

    for i, epoch in enumerate(range(first_epoch, last_epoch + 1)):
        if epoch == first_epoch:
            repeat = LEARN_REP_NUM * 10
        else:
            repeat = LEARN_REP_NUM

        lr = lr_schedule[-1][0]
        for lr_change_index, lr_cand in lr_schedule[::-1]:
            if i < lr_change_index:
                lr = lr_cand

        loss_dict, valid_loss = AI.learn(epoch, repeat, loss_dict, valid_loss, long_warmup_epoch_num=first_epoch, data_dir=data_path, learning_rate=lr)  # long_warmup_epoch_num=first_epochとすることで最初のepochだけ全部warmupにすることができる。
        AI.save(os.path.join(ckpt_path, "parameter/train_experiment.ckpt"))

    print("")
    elapsed = time.time() - s
    print(f"train time = {elapsed} [s]")


def train_without_selfplay_process(x):
    epoch = 800
    path = "train_results"

    AI = CNNAI(0)
    AI.load(os.path.join(path, f"parameter/epoch{epoch - 1}.ckpt"))
    
    repeat = 1
    #repeat = LEARN_REP_NUM
    print(f"target epoch = {epoch}, repeat num = {repeat}")
    s = time.time()
    AI.learn(epoch, repeat)
    #AI.save(os.path.join(path, "parameter/post.ckpt"))
    print("")
    elapsed = time.time() - s
    print(f"train time = {elapsed} [s]")


def measure_inference_time(x):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPUで実行時間計測ならコメントを外す

    filters = 32
    layer_num = 9

    # 初期パラメータで学習データに対して推論を実施してみて実行時間を計測
    #AI = CNNAI(0, filters=filters, layer_num=layer_num, use_global_pooling=False)
    AI = CNNAI(0)
    #AI.load("data_for_experiment/221204/train_results/parameter/epoch30.ckpt")

    h5file = h5py.File("data_for_experiment/60/6000.h5", "r")
    size = h5file["feature"].shape[0]
    feature_arr = h5file["feature"][:, :, :, :]

    s = time.time()
    INFERENCE_NUM = 1000 # MCTS中でpvの計算にかかる時間の測定という想定。大きくしすぎてinput_arrのindexを飛び出さないよう注意
    total = 0
    for i in tqdm(range(INFERENCE_NUM)):
        input_arr = feature_arr[(i % 50) * AI.n_parallel: ((i % 50) + 1) * AI.n_parallel]
        #input_arr = np.random.rand(*input_arr.shape)
        #p = AI.sess.run(AI.p_tf, feed_dict={AI.x:input_arr})
        p, y_pred = AI.sess.run([AI.p_tf, AI.y], feed_dict={AI.x:input_arr})
        total += np.sum(p) + np.sum(y_pred)  # p, y_predの計算をサボっていないか確認するため
    print(total)

    elapsed = time.time() - s
    print("{} inference time = {:.2f} [s]".format(INFERENCE_NUM, elapsed))
    print("1 inference time = {:.2f} [ms]".format(elapsed / INFERENCE_NUM * 1000))


def train_without_selfplay():
    print("="*30)
    with Pool(processes=1) as p:
        p.map(func=measure_inference_time, iterable=[0])

    exit()

    print("="*30)
    with Pool(processes=1) as p:
        p.map(func=train_from_random_parameter_process, iterable=[0])

    restart_filename = "train_results/restart230925.pkl"  # このrestartyymmdd.pklは手動でコピーして作る
    initial_epoch, AI_id_list, AI_rate_list, new_rate, load_AI_id, loss_dict, valid_loss = pickle.load(open(restart_filename, "rb"))
    rate, _ = evaluate_and_calc_rate(AI_id_list, AI_rate_list, "train_experiment.ckpt")
    print("")
    print("レート{:.4f}".format(rate))


def generate_h5(h5_id, display, AI_id, search_nodes, epoch, is_test=False):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    seed = h5_id * 10000 % (2**30)
    random.seed(seed)
    is_random_AI = (AI_id == 0)
    #print(h5_id, display, AI_id, search_nodes, is_random_AI)

    if is_random_AI:
        AIs = [CNNAI(0, search_nodes=1, all_parameter_zero=True, p_is_almost_flat=True, seed=seed, random_playouts=False),
               CNNAI(1, search_nodes=1, all_parameter_zero=True, p_is_almost_flat=True, seed=seed, random_playouts=False)]
        # AIs = [CNNAI(0, search_nodes=search_nodes, all_parameter_zero=True, p_is_almost_flat=True, seed=h5_id*10000),
        #        CNNAI(1, search_nodes=search_nodes, all_parameter_zero=True, p_is_almost_flat=True, seed=h5_id*10000)]
    else:
        AIs = [CNNAI(0, search_nodes=search_nodes, seed=h5_id*10000 % (2**30), random_playouts=True),
               CNNAI(1, search_nodes=search_nodes, seed=h5_id*10000 % (2**30), random_playouts=True)]
        AIs[0].load(os.path.join(PARAMETER_DIR, "epoch{}.ckpt".format(AI_id)))
        AIs[1].load(os.path.join(PARAMETER_DIR, "epoch{}.ckpt".format(AI_id)))

    b_win = 0
    w_win = 0
    draw = 0
    turn_list = []

    all_data = []
    for _ in range(GAME_NUM_IN_H5):
        is_mimic_AI = False
        force_opening = None
        if random.random() < MIMIC_AI_RATIO:
            is_mimic_AI = True
        elif MIMIC_AI_RATIO <= random.random() < MIMIC_AI_RATIO + FORCE_OPENING_RATE:
            force_opening = random.choice(FORCE_OPENING_LIST)
        AIs[0].force_opening = force_opening
        AIs[1].is_mimic_AI = is_mimic_AI
        AIs[1].force_opening = force_opening

        if h5_id < H5_NUM:
            temp_data = generate_data(AIs, 1, noise=1., id_=h5_id, display=is_test, info=is_test)
        else:
            # display = False
            # if h5_id == 326919:
            #     display = True
            temp_data = generate_data(AIs, 1, noise=NOISE, id_=h5_id, display=is_test, info=is_test)
        all_data.extend(temp_data)

        if len(temp_data) == 0:
            draw += 1
        else:
            turn_list.append(len(temp_data) // 4)  # データ数からターン数を逆算しているため、途中でデータ生成を打ち切るようになると実際のターン数とは一致しない
            reward = -temp_data[-1][2]  # augmentationの影響で報酬がひっくり返っている
            if reward == 1:
                b_win += 1
            elif reward == -1:
                w_win += 1

    data_size = len(all_data)
    # ***h5に項目を追加したら、shape_listにshapeを追加する必要がある***
    shape_list = [(data_size, 9, 9, CHANNEL), (data_size, 137), (data_size,), (data_size,), (data_size,), (data_size,), 
                  (data_size,), (data_size,), (data_size,), (data_size,), (data_size,), (data_size,),
                  (data_size, 8, 8), (data_size, 8, 8), (data_size, 9, 9), (data_size, 9, 9), (data_size, 9, 9), (data_size, 9, 9), (data_size, 137)]
    data_list = [np.zeros(shape) for shape in shape_list]
    
    for i, each_data in enumerate(all_data):
        assert len(data_list) == len(each_data)
        for j in range(len(data_list)):
            data_list[j][i] = each_data[j]

    del all_data

    # if display:
    #     sys.stderr.write('\r\033[Kepoch{}:data={}/{} B{}win W{}win {}draw'.format(epoch + 1, index, STEP, b_win, w_win, draw))
    #     sys.stderr.flush()

    if not is_test:
        h5file = h5py.File(os.path.join(DATA_DIR, str(epoch), "{}.h5".format(h5_id)), "w")
        for h5_name, data in zip(H5_NAME_LIST, data_list):
            h5file.create_dataset(h5_name, data=data, compression="gzip", compression_opts=1)
        h5file.flush()
        h5file.close()

    del AIs
    return b_win, w_win, draw, sum(turn_list) / len(turn_list)


def generate_h5_single(pair):
    h5_id_, display, AI_id, search_nodes, epoch, wait_time = pair
    time.sleep(wait_time)
    return generate_h5(h5_id_, display, AI_id, search_nodes, epoch)


def train_AIs_process(arg):
    epoch, loss_dict, valid_loss, save_epoch = arg
        
    AI = CNNAI(0, search_nodes=search_nodes, per_process_gpu_memory_fraction=0.5)

    long_warmup_epoch_num = POOL_EPOCH_NUM
    #long_warmup_epoch_num = 250

    if epoch <= long_warmup_epoch_num:
        pass
        #AI.save(os.path.join(PARAMETER_DIR, "epoch0.ckpt"))
    else:
        AI.load(os.path.join(PARAMETER_DIR, "post.ckpt"))


    if epoch <= long_warmup_epoch_num:
        learn_rep_num = LEARN_REP_NUM * 10  # warmup用
        ret = AI.learn(epoch, learn_rep_num, loss_dict, valid_loss, long_warmup_epoch_num=long_warmup_epoch_num)
    else:
        learn_rep_num = LEARN_REP_NUM
        ret = AI.learn(epoch, learn_rep_num, loss_dict, valid_loss)
    
    if save_epoch:
        AI.save(os.path.join(PARAMETER_DIR, "epoch{}.ckpt".format(epoch)))
    AI.save(os.path.join(PARAMETER_DIR, "post.ckpt"))

    del AI
    return ret


def evaluate_2game_process(arg_tuple):
    # 先後で２試合して勝利数を返す
    AI_id, _, seed, AI_load_name = arg_tuple

    if AI_id == -1:
        AI1 = CNNAI(0, search_nodes=EVALUATION_SEARCHNODES, all_parameter_zero=True, p_is_almost_flat=True, seed=seed)
    else:
        AI1 = CNNAI(0, search_nodes=EVALUATION_SEARCHNODES, seed=seed)
        AI1.load(os.path.join(PARAMETER_DIR, "epoch{}.ckpt".format(AI_id)))

    AIs = [AI1, CNNAI(1, search_nodes=EVALUATION_SEARCHNODES, seed=seed)]
    AIs[0].tau = EVALUATION_TAU
    AIs[1].tau = EVALUATION_TAU

    AIs[1].load(os.path.join(PARAMETER_DIR, AI_load_name.format(AI_id)))

    ret = evaluate(AIs, 2, multiprocess=True)
    del AIs
    return ret


def evaluate_and_calc_rate(AI_id_list, AI_rate_list, AI_load_name="post.ckpt", evaluate_play_num=EVALUATE_PLAY_NUM):
    play_num = evaluate_play_num // len(AI_id_list) // 2 * 2
    win_num_list = []
    for old_AI_id, old_rate in zip(AI_id_list, AI_rate_list):
        play_num_half = play_num // 2
        # for x in tqdm([(old_AI_id, search_nodes, j * 10000, AI_load_name) for j in range(play_num_half)]):
        #     evaluate_2game_process(x)
        with Pool(processes=PROCESS_NUM) as p:
            imap = p.imap(func=evaluate_2game_process, iterable=[(old_AI_id, search_nodes, j * 10000 % (2**30), AI_load_name) for j in range(play_num_half)])
            ret = list(tqdm(imap, total=play_num_half, file=sys.stdout))
        new_ai_win_num = play_num - sum(ret)
        win_num_list.append(new_ai_win_num)
        print("new AI vs {} (rate={:.3f}) : {}/{}, win rate={:.3f}".format(old_AI_id, old_rate, new_ai_win_num, play_num, new_ai_win_num / play_num))
    
    return calc_rate(play_num, np.array(AI_rate_list), np.array(win_num_list)), win_num_list


def draw_all_graphs():
    df = pd.read_csv(os.path.join(TRAIN_LOG_DIR, "train_log.csv"))

    time_sum_list = []
    time_sum = 0.0
    prev_past_epoch = df["epoch"].values[0] - 1
    for past_epoch, dt in zip(df["epoch"], df["train cycle time [s]"]):
        time_sum += (past_epoch - prev_past_epoch) * dt / 86400
        time_sum_list.append(time_sum)
        prev_past_epoch = past_epoch

    plt.clf()
    plt.plot(time_sum_list, df["data generation AI rate"], label="data generation AI rate")
    plt.xlabel("train time [day]")
    plt.legend()
    plt.savefig(os.path.join(TRAIN_LOG_DIR, "data_generation_AI_rate.png"))

    plt.clf()
    plt.plot(df["epoch"], df["game num per day [/day]"], label="game num per day [/day]")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(TRAIN_LOG_DIR, "game_generation_speed.png"))

    plt.clf()
    plt.plot(df["epoch"], df["b win num"] / df["game num"], label="sente win rate")
    plt.plot(df["epoch"], df["w win num"] / df["game num"], label="gote win rate")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(TRAIN_LOG_DIR, "bw_win_rate.png"))

    plt.clf()
    plt.plot(df["epoch"], df["draw num"] / df["game num"], label="draw rate")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(TRAIN_LOG_DIR, "draw_rate.png"))

    plt.clf()
    plt.plot(df["epoch"], df["turn average"], label="turn average")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(TRAIN_LOG_DIR, "turn_average.png"))

    plt.clf()
    plt.plot(df["epoch"], df["train loss ema"], label="train loss ema")
    plt.plot(df["epoch"], df["valid loss ema"], label="valid loss ema")
    losses = np.array(list(df["train loss ema"]) + list(df["valid loss ema"]))
    #losses = sorted(losses)
    #plt.ylim(top=max(3.5, losses[int(len(losses) * 0.9)] + 0.1))
    plt.ylim(top=np.percentile(losses, 90) + 0.1, bottom=np.percentile(losses, 10) - 0.1)
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(os.path.join(TRAIN_LOG_DIR, "total_loss_ema.png"))

    y_plot_name_list = ["v loss ema", "vreg loss ema", "p loss ema", "preg loss ema"]
    for y_plot_name in y_plot_name_list:
        plot_filename = y_plot_name.replace(" ", "_") + ".png"
        plt.clf()
        plt.plot(df["epoch"], df[y_plot_name], label=y_plot_name)
        values = df[y_plot_name].values
        plt.ylim(top=np.percentile(values, 90) + 0.1, bottom=np.percentile(values, 10) - 0.1)
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(os.path.join(TRAIN_LOG_DIR, plot_filename))

    for name in AUX_NAME_LIST:
        plt.clf()
        plt.plot(df["epoch"], df[name], label=name)
        values = df[name].values
        scale = np.max(values) - np.min(values)
        plt.ylim(top=np.percentile(values, 90) + scale / 10, bottom=np.percentile(values, 10) - scale / 10)
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(os.path.join(TRAIN_LOG_DIR, f"{name}_loss.png"))


def learn(search_nodes, restart=False, skip_first_selfplay=False, restart_filename="train_results/restart.pkl"):
    initial_epoch = 0
    load_AI_id = 0
    
    AI_id_list = [-1]  # レート測定のためのAIの番号リスト。-1は初期AIを表す。
    AI_rate_list = [0.0]
    prev_rate = new_rate = 0.0  # 学習に使っているAIのレート

    loss_dict = {"train_loss": 5.0, "v_loss": 1.0, "v_reg_loss": 1.0, "p_loss": 5.0, "p_reg_loss": 5.0}
    for name in AUX_NAME_LIST:
        loss_dict[name] = 5.0
    valid_loss = 5.0

    if restart:
        initial_epoch, AI_id_list, AI_rate_list, new_rate, load_AI_id, loss_dict, valid_loss = pickle.load(open(restart_filename, "rb"))
        print("retrain: {}, {}, {}, {}, {}, {}".format(initial_epoch, AI_id_list, AI_rate_list, new_rate, load_AI_id, loss_dict, valid_loss))

    print("epoch num in pool = {}".format(POOL_EPOCH_NUM))
    epoch = initial_epoch
    while True:
        epoch_start_time = time.time()

        # -------data generation-------
        start = time.time()

        print("=" * 50)
        print(f"epoch {epoch}: AI{load_AI_id} self-playing")
        
        # 生成されていないh5のリストを求める
        os.makedirs(os.path.join(DATA_DIR, str(epoch)), exist_ok=True)
        h5_set = set(range(epoch * EPOCH_H5_NUM, (epoch + 1) * EPOCH_H5_NUM))
        files = os.listdir(os.path.join(DATA_DIR, str(epoch)))
        h5_exist_set = set([int(s.split(".")[0]) for s in files])
        h5_list = list(h5_set - h5_exist_set)
        h5_list = sorted(h5_list)
        wait_time_list = [(x - min(h5_list)) * 3 if x - min(h5_list) < PROCESS_NUM else 0 for x in h5_list]

        with Pool(processes=PROCESS_NUM) as p:
            imap = p.imap(func=generate_h5_single, iterable=[(h5_id_, h5_id_ % PROCESS_NUM == 0, load_AI_id, search_nodes, epoch, wait_time) for h5_id_, wait_time in zip(h5_list, wait_time_list)])
            win_tuple_list = list(tqdm(imap, total=len(h5_list), file=sys.stdout))
        
        print("")
        #print(win_tuple_list)

        elapsed_time = time.time() - start
        print("elapsed time(self-play)={}".format(elapsed_time))

        epoch += 1
        if epoch < POOL_EPOCH_NUM:
            continue

        if not USE_EVALUATION_RESULT:
            load_AI_id = epoch
        
        # --------training---------
        start = time.time()
    
        # 親プロセスでtfを起動すると子プロセスでエラーが出るのであえて子プロセスに分けて学習する
        with Pool(processes=1) as p:
            ret = p.map(func=train_AIs_process, iterable=[(epoch, loss_dict, valid_loss, True)])
        loss_dict, valid_loss = ret[0]

        print("")
        print("elapsed time(training)={}".format(time.time() - start))
        
        # restart用保存
        pickle.dump((epoch, AI_id_list, AI_rate_list, new_rate, load_AI_id, loss_dict, valid_loss), open(restart_filename, "wb"))

        # --------evaluation---------
        if epoch % EVALUATION_EPOCH_NUM == 0:
            start = time.time()
            new_rate, win_num_list = evaluate_and_calc_rate(AI_id_list, AI_rate_list)

            win_num_dir = os.path.join(TRAIN_LOG_DIR, "win_num")
            os.makedirs(win_num_dir, exist_ok=True)
            np.savetxt(os.path.join(win_num_dir, f"epoch{epoch}_win_num.csv"), np.array(win_num_list), delimiter=',')

            if USE_EVALUATION_RESULT:
                if new_rate >= prev_rate + RATE_TH:
                    prev_rate = new_rate
                    load_AI_id = epoch
                    print("new AI accepted. rate={}".format(new_rate))
                else:
                    print("new AI rejected. new rate={}, prev rate={}".format(new_rate, prev_rate))
            else:
                print("evaluation done. new rate={}, prev AI_rate_list={}".format(new_rate, AI_rate_list))

            # 評価用AIの更新
            if new_rate >= AI_rate_list[-1] + RATE_TH2:
                if len(AI_id_list) >= MAX_AI_NUM_FOR_EVALUATE:
                    AI_id_list.pop(0)
                    AI_rate_list.pop(0)
                AI_id_list.append(epoch)
                AI_rate_list.append(new_rate)
                print("new AI list")
                print(AI_id_list, AI_rate_list)
                np.savetxt(os.path.join(win_num_dir, f"epoch{epoch}_AI_id_list.csv"), np.array(AI_id_list), delimiter=',')
                np.savetxt(os.path.join(win_num_dir, f"epoch{epoch}_AI_rate.csv"), np.array(AI_rate_list), delimiter=',')

            print("elapsed time(evaluation)={}".format(time.time() - start))
            print("")

        gc.collect()

        epoch_elapsed_time = time.time() - epoch_start_time

        # restart用保存
        pickle.dump((epoch, AI_id_list, AI_rate_list, new_rate, load_AI_id, loss_dict, valid_loss), open(restart_filename, "wb"))

        # 棋譜化して、参照しないh5は削除する
        with Pool(processes=1) as p:
            ret = p.map(func=analyze_h5_main, iterable=[()])
        existing_data_dir_list = sorted([int(x) for x in os.listdir(DATA_DIR)])
        now_epoch = max(existing_data_dir_list)
        for data_dir in existing_data_dir_list:
            rm_dir = os.path.join(DATA_DIR, str(data_dir))
            if data_dir < now_epoch - SAVE_H5_NUM:
                print("deleted", rm_dir)
                shutil.rmtree(rm_dir)

        if len(win_tuple_list) == EPOCH_H5_NUM:
            write_dict = OrderedDict()
            
            write_dict["datetime"] = str(datetime.datetime.today())
            write_dict["epoch"] = int(epoch)
            write_dict["data generation AI rate"] = float(new_rate) 
            write_dict["train cycle time [s]"] = float(epoch_elapsed_time) 
            write_dict["b win num"] = sum([x[0] for x in win_tuple_list])
            write_dict["w win num"] = sum([x[1] for x in win_tuple_list])
            write_dict["draw num"] = sum([x[2] for x in win_tuple_list])
            game_num = sum([x[0] for x in win_tuple_list]) + sum([x[1] for x in win_tuple_list]) + sum([x[2] for x in win_tuple_list])
            write_dict["game num"] = game_num
            write_dict["game num per day [/day]"] = game_num / epoch_elapsed_time * 86400
            write_dict["turn average"] = sum([x[3] for x in win_tuple_list]) / len([x[3] for x in win_tuple_list])
            write_dict["train loss ema"] = float(loss_dict["train_loss"])
            write_dict["valid loss ema"] = float(valid_loss)
            write_dict["v loss ema"] = loss_dict["v_loss"]
            write_dict["vreg loss ema"] = loss_dict["v_reg_loss"]
            write_dict["p loss ema"] = loss_dict["p_loss"]
            write_dict["preg loss ema"] = loss_dict["p_reg_loss"]
            for name in AUX_NAME_LIST:
                write_dict[name] = loss_dict[name]

            # 初回はheadを書き込む
            if epoch == POOL_EPOCH_NUM:
                with open(os.path.join(TRAIN_LOG_DIR, "train_log.csv"), "w") as file:
                    writer = csv.writer(file)
                    writer.writerow(list(write_dict.keys()))

            # 値の書き込み
            with open(os.path.join(TRAIN_LOG_DIR, "train_log.csv"), "a") as file:
                writer = csv.writer(file)
                writer.writerow(list(write_dict.values()))
                # file.write("{},{},{},{},{},{}".format(datetime.datetime.today(),
                #                                 #check_output(["git", "show", '--format="%H"', "-s"]).rstrip(),
                #                                 i, new_rate, epoch_elapsed_time, STEP * (next_training_epoch - old_i) / epoch_elapsed_time, trainema) + os.linesep)
        
            draw_all_graphs()



if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PARAMETER_DIR, exist_ok=True)
    os.makedirs(TRAIN_LOG_DIR, exist_ok=True)

    np.seterr(divide='raise', invalid='raise')
    search_nodes = SELFPLAY_SEARCHNODES_MIN
    if len(sys.argv) >= 3:
        search_nodes = int(sys.argv[2])

    if sys.argv[1] == "train":
        learn(search_nodes)
    elif sys.argv[1] == "retrain":
        learn(search_nodes, restart=True, skip_first_selfplay=False)
    elif sys.argv[1] == "train_without_selfplay":
        # 既にあるselfplayデータを使って過学習などについて解析をする
        train_without_selfplay()
    elif sys.argv[1] == "debug_learn":
        # １つ目のAIを距離だけで考えるものとして、そこからどれだけ学習の結果強くなれるかを検証する
        debug_learn([CNNAI(0, search_nodes=search_nodes, v_is_dist=True, p_is_almost_flat=True), CNNAI(1, search_nodes=search_nodes, v_is_dist=True)])
    elif sys.argv[1] == "view":
        AIs = [CNNAI(0, search_nodes=search_nodes, tau=0.25, seed=100), CNNAI(1, search_nodes=search_nodes, tau=0.25, seed=100)]
        #AIs = [CNNAI(0, search_nodes=search_nodes, tau=0.5, seed=100), CNNAI(1, search_nodes=search_nodes, tau=0.5, seed=100, is_mimic_AI=True)]
        # AIs[0].load(os.path.join(PARAMETER_DIR, "train_experiment.ckpt"))
        # AIs[1].load(os.path.join(PARAMETER_DIR, "train_experiment.ckpt"))
        AIs[0].load(os.path.join(PARAMETER_DIR, "epoch3090.ckpt"))
        AIs[1].load(os.path.join(PARAMETER_DIR, "epoch3090.ckpt"))

        # AIs = [CNNAI(0, search_nodes=search_nodes, tau=0.25, all_parameter_zero=True, v_is_dist=True, p_is_almost_flat=True), 
        #        CNNAI(1, search_nodes=search_nodes, tau=0.25, all_parameter_zero=True, v_is_dist=True, p_is_almost_flat=True)]

        # search_nodes = 1
        # AIs = [CNNAI(0, search_nodes=search_nodes, tau=0.25, all_parameter_zero=True, p_is_almost_flat=True), 
        #        CNNAI(1, search_nodes=search_nodes, tau=0.25, all_parameter_zero=True, p_is_almost_flat=True)]

        #normal_play(AIs)
        game_num = 10
        for i in range(game_num):
            AIs[0].init_prev()
            AIs[1].init_prev()
            normal_play(AIs)
            
            if i < game_num - 1:
                print("input key to see next game")
                input()

    elif sys.argv[1] == "vs":
        AIs = [Human(0), CNNAI(1, search_nodes=search_nodes, tau=0.5)]
        AIs[1].load(os.path.join(PARAMETER_DIR, "post.ckpt"))
        normal_play(AIs)
    elif sys.argv[1] == "test":
        #np.random.seed(0)
        AIs = [CNNAI(0, search_nodes=search_nodes, tau=0.5), CNNAI(1, search_nodes=search_nodes, tau=0.5)]
        AIs[0].load(os.path.join(PARAMETER_DIR, "post.ckpt"))
        AIs[1].load(os.path.join(PARAMETER_DIR, "post.ckpt"))
        for i in range(1):
            print("============={}==============".format(i))
            normal_play(AIs)
    elif sys.argv[1] == "evaluate":
        def evaluate_2game_process(seed):
            # 先後で２試合して勝利数を返す
            AIs = [CNNAI(0, search_nodes=search_nodes, tau=0.5, seed=seed), CNNAI(1, search_nodes=search_nodes, tau=0.5, seed=seed, use_average_Q=True)]
            #AIs[0].load(os.path.join("backup/221219/train_results/parameter/", "epoch680.ckpt"))
            AIs[0].load(os.path.join(PARAMETER_DIR, "epoch120.ckpt"))
            AIs[1].load(os.path.join(PARAMETER_DIR, "epoch120.ckpt"))
            ret = evaluate(AIs, 2, multiprocess=True, display=True)
            del AIs
            return ret
        play_num = 4
        play_num_half = play_num // 2
        with Pool(processes=1) as p:
            imap = p.imap(func=evaluate_2game_process, iterable=[j * 10000 % (2**30) for j in range(play_num_half)])
            ret = list(tqdm(imap, total=play_num_half, file=sys.stdout))
        print(sum(ret), play_num - sum(ret))
        
    elif sys.argv[1] == "multiprocess_test":
        np.random.seed(0)
        all_game_num = 100
        game_num = 2
        def generate_data_single(seed):
            AIs = [CNNAI(0, search_nodes=search_nodes, seed=seed, tau=0.5),
                CNNAI(1, search_nodes=search_nodes, seed=seed, tau=0.5)]
            path = os.path.join(PARAMETER_DIR, "epoch290.ckpt")
            AIs[0].load(path)
            AIs[1].load(path)
            temp_data = generate_data(AIs, game_num, noise=NOISE)
            return len(temp_data)
        task_num = all_game_num // game_num
        start = time.time()
        with Pool(processes=PROCESS_NUM) as p:
            imap = p.imap(func=generate_data_single, iterable=range(task_num))
            turn_list = list(tqdm(imap, total=task_num, file=sys.stdout))
        print(turn_list)
        print(sum(turn_list))
        print("elapsed time = {:.3f}s".format(time.time() - start))

    elif sys.argv[1] == "generate_h5_test":
        epoch = 4000
        h5_id = epoch * EPOCH_H5_NUM
        AI_id = epoch
        generate_h5(h5_id, True, AI_id, search_nodes, epoch, is_test=True)

    elif sys.argv[1] == "measure":
        np.random.seed(0)
        game_num = 10
        seed = 0
        AIs = [CNNAI(0, search_nodes=search_nodes, seed=seed, random_playouts=True), CNNAI(1, search_nodes=search_nodes, seed=seed, random_playouts=True)]
        path = PARAMETER_DIR
        AIs[0].load(os.path.join(path, "epoch240.ckpt"))
        AIs[1].load(os.path.join(path, "epoch240.ckpt"))
        # AIs[0].load(os.path.join(PARAMETER_DIR, "epoch4175.ckpt"))
        # AIs[1].load(os.path.join(PARAMETER_DIR, "epoch4175.ckpt"))

        start_time = time.time()
        temp_data = generate_data(AIs, game_num, noise=NOISE, display=True, info=True)
        # with open("measure.log", "a") as f:
        #     f.write("{},{},{}".format(datetime.datetime.today(),
        #                               #check_output(["git", "show", '--format="%H"', "-s"]).rstrip(),
        #                               time.time() - start_time, game_num) + os.linesep)
        print()
        print("{:.2f}s".format(time.time() - start_time))

    elif sys.argv[1] == "measure_cpu_time":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        np.random.seed(0)
        game_num = 2
        seed = 0
        search_nodes = 500
        AIs = [CNNAI(0, search_nodes=search_nodes, seed=seed, tau=0.32),
               CNNAI(1, search_nodes=search_nodes, seed=seed, tau=0.32)]
        path = PARAMETER_DIR
        epoch = 240

        AIs[0].load(os.path.join("application_data/parameter", "epoch{}.ckpt".format(epoch)))
        AIs[1].load(os.path.join("application_data/parameter", "epoch{}.ckpt".format(epoch)))

        start_time = time.time()
        temp_data = evaluate(AIs, game_num, display=True)
        print("{:.2f}s".format(time.time() - start_time))

    elif sys.argv[1] == "train_from_existing_data":
        # モデルを大きくしたとき等に良いパラメータの初期値を得るため、自己対戦データを順に学習していく
        first_epoch = 250
        last_epoch = 297

        loss_dict = {"train_loss": 5.0, "v_loss": 1.0, "v_reg_loss": 1.0, "p_loss": 5.0, "p_reg_loss": 5.0}
        valid_loss = 5.0
        for name in AUX_NAME_LIST:
            loss_dict[name] = 5.0

        for epoch in range(first_epoch, last_epoch + 1):
            with Pool(processes=1) as p:
                ret = p.map(func=train_AIs_process, iterable=[(epoch, loss_dict, valid_loss, True)])
            loss_dict, valid_loss = ret[0]
