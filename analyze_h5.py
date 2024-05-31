# coding:utf-8
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
from State import RIGHT, DOWN, State, BOARD_LEN, State_init, display_cui
import pickle
import argparse
from BasicAI import display_parameter
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # warning抑制？
LOAD_GAME_LIMIT = 10

PLACEWALL_DICT_KEYS = ["<=-1", "0", "1", "2", "3", "4", ">=5"]

def calc_entropy(feature_arr):
    size = feature_arr.shape[0]
    feature_tuple_list = [tuple(x.flatten()) for x in feature_arr]

    feature_counter = Counter(feature_tuple_list)
    num_sorted = np.array([x[1] for x in feature_counter.most_common()])
    num_dist = num_sorted / size
    #print(num_sorted[:100])
    #print(num_dist[:30])
    #print("max entropy = {}".format(np.log(size)))
    return np.sum(-num_dist * np.log(num_dist))


def get_pos(feature):
    B_pos_arr = feature[:, :, 0]
    W_pos_arr = feature[:, :, 1]
    B_index = np.argmax(B_pos_arr)
    W_index = np.argmax(W_pos_arr)
    Bx = B_index // 9
    By = B_index % 9
    Wx = W_index // 9
    Wy = W_index % 9
    return Bx, By, Wx, Wy


def divide_arr_to_games(epoch_arr_dict, ignore_augmented_data=True):
    # ゴール到着までデータが存在すると仮定し、コマが飛ぶことを検知してデータを試合単位に分割する
    #for i in range(epoch_feature_arr.shape[0]):
    
    if ignore_augmented_data:
        # データ拡張したデータを無視する
        main_index_lists = [[i * 4 for i in range(epoch_arr_dict["feature"].shape[0] // 4)]]
    else:
        main_index_lists = []
        for k in range(4):
            main_index_lists.append([i * 4 + k for i in range(epoch_arr_dict["feature"].shape[0] // 4)])

    ret = None

    for main_index_i, main_index_list in enumerate(main_index_lists):
        main_feature_arr = epoch_arr_dict["feature"][main_index_list]

        prev_By = 8
        prev_Wy = 0
        div_index_list = [0]
        for i in range(main_feature_arr.shape[0]):
            feature = main_feature_arr[i]
            Bx, By, Wx, Wy = get_pos(feature)
            if abs(By - prev_By) >= 3 or abs(Wy - prev_Wy) >= 3 :
                div_index_list.append(i)
            prev_By = By
            prev_Wy = Wy
        div_index_list.append(main_feature_arr.shape[0])

        if ret is None:
            ret = {}
            for x_name in H5_NAME_LIST:
                ret[x_name] = [None] * (len(div_index_list) - 1) * len(main_index_lists)  # 試合数*データ拡張による複製数

        for x_name in H5_NAME_LIST:
            for game_i, s, e in zip(range(len(div_index_list[:-1])), div_index_list[:-1], div_index_list[1:]):
                ret[x_name][game_i * len(main_index_lists) + main_index_i] = epoch_arr_dict[x_name][main_index_list][s:e]

    return ret


def analyze_placewall(arr_per_game_list_dict):
    feature_arr_list = arr_per_game_list_dict["feature"]
    v_post_arr_list = arr_per_game_list_dict["v_post"]
    
    placewall_distdiff_list = []
    determine_wall_count = 0
    for feature_arr, v_post_arr in zip(feature_arr_list, v_post_arr_list):
        prev_Bx = 4
        prev_By = 8
        prev_Wx = 4
        prev_Wy = 0

        B_prev_distdiff = 0
        W_prev_distdiff = 0
        array_size = feature_arr.shape[0]
        for i in range(array_size):
            feature = feature_arr[i]

            next_v_post = None
            if i < array_size - 1:
                next_v_post = v_post_arr[i + 1]

            Bx, By, Wx, Wy = get_pos(feature)
            is_placewall = (Bx == prev_Bx) and (By == prev_By) and (Wx == prev_Wx) and (Wy == prev_Wy)

            is_black = i % 2

            if is_black:
                my_dist = feature[Bx, By, 9] * 20  # 特徴量の時点で/20されている
                opponent_dist = feature[Wx, Wy, 10] * 20
            else:
                my_dist = feature[Wx, Wy, 10] * 20  # 特徴量の時点で/20されている
                opponent_dist = feature[Bx, By, 9] * 20
            distdiff = opponent_dist - my_dist

            if is_black:
                placewall_distdiff = distdiff - B_prev_distdiff
                
            else:
                placewall_distdiff = distdiff - W_prev_distdiff
                W_prev_distdiff = distdiff

            #print(Bx, By, Wx, Wy, is_placewall)
            #print(my_dist, opponent_dist, opponent_dist - my_dist, placewall_distdiff)

            if is_placewall:
                placewall_distdiff_list.append(placewall_distdiff)

                # end gameのときv_post=0になるのを利用して確定路end gameになる壁置きをカウントする
                black_walls = int(feature[0, 0, 2] * 10)
                white_walls = int(feature[0, 0, 3] * 10)
                if next_v_post is not None and next_v_post == 0.0 and black_walls >= 1 and white_walls >= 1:
                    determine_wall_count += 1

            prev_Bx = Bx
            prev_By = By
            prev_Wx = Wx
            prev_Wy = Wy
            if is_black:
                B_prev_distdiff = distdiff
                W_prev_distdiff = -distdiff
            else:
                B_prev_distdiff = -distdiff
                W_prev_distdiff = distdiff

    counter = Counter(placewall_distdiff_list)

    ret = {}
    for key in PLACEWALL_DICT_KEYS:
        ret[key] = 0
    for distdiff, count in counter.items():
        if distdiff <= -1:
            ret["<=-1"] += count
        elif distdiff >= 5:
            ret[">=5"] += count
        else:
            ret[str(int(distdiff))] = count

    return ret, determine_wall_count


def get_state_from_feature(feature, turn):
    Bx, By, Wx, Wy = get_pos(feature)
    cross_arr = np.array(feature[:, :, 5:9], dtype=bool)
    column_wall = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=bool)
    row_wall = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=bool)

    for x in range(BOARD_LEN - 1):
        after = False
        for y in range(BOARD_LEN - 1):
            if not cross_arr[x, y, RIGHT] and not after:
                column_wall[x, y] = True
                after = True
            else:
                after = False
    for y in range(BOARD_LEN - 1):
        after = False
        for x in range(BOARD_LEN - 1):
            if not cross_arr[x, y, DOWN] and not after:
                row_wall[x, y] = True
                after = True
            else:
                after = False

    state = State()
    State_init(state)
    state.black_walls = state.state_c.black_walls = int(feature[0, 0, 2] * 10)
    state.white_walls = state.state_c.white_walls = int(feature[0, 0, 3] * 10)
    state.turn = state.state_c.turn = turn
    state.Bx = state.state_c.Bx = Bx
    state.By = state.state_c.By = By
    state.Wx = state.state_c.Wx = Wx
    state.Wy = state.state_c.Wy = Wy
    state.column_wall = column_wall
    state.row_wall = row_wall
    return state


def get_kifu_from_features(features):
    states = []
    for turn, feature in enumerate(features):
        states.append(get_state_from_feature(feature, turn))

    ret = []
    prev_state = states[0]
    for state in states[1:]:
        is_placewall = (state.Bx == prev_state.Bx) and (state.Wx == prev_state.Wx) and (state.By == prev_state.By) and (state.Wy == prev_state.Wy)

        if is_placewall:
            row_wall_diff = state.row_wall & ~prev_state.row_wall
            column_wall_diff = state.column_wall & ~prev_state.column_wall
            
            if np.max(row_wall_diff) > 0.5:
                wall_str = "h"
                wall_diff = row_wall_diff
            else:
                wall_str = "v"
                wall_diff = column_wall_diff

            argmax = np.argmax(wall_diff)

            wall_x = argmax // 8
            wall_y = argmax % 8

            # print(wall_str)
            # print(wall_diff)
            # print(argmax)
            # print(wall_x, wall_y)
            
            action_str = chr(wall_x + 97) + str(wall_y + 1) + wall_str
        else:
            is_black = (prev_state.turn % 2 == 0)
            if is_black:
                x = state.Bx
                y = state.By
            else:
                x = state.Wx
                y = state.Wy
            action_str = chr(x + 97) + str(y + 1)

        ret.append(action_str)

        prev_state = state
    
    return ret


def save_all_kifu(arr_per_game_list_dict, each_data_dir, div_i):
    features_list = arr_per_game_list_dict["feature"]
    save_str = ""
    for features in features_list:
        kifu = get_kifu_from_features(features)
        save_str += ",".join(kifu) + os.linesep
    save_dir = os.path.join(KIFU_DIR, str(each_data_dir))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{div_i}.txt"), "w") as fout:
        fout.write(save_str)


def display_state_from_feature(feature, turn, output_array=False):
    state = get_state_from_feature(feature, turn)
    display_cui(state, check_algo=False)

    if output_array:
        output_dir = "DP_testcase/1"
        os.makedirs(output_dir, exist_ok=True)
        #pickle.dump(state, open(os.path.join(output_dir, f"{turn}_state.pkl"), "wb"))
        np.savetxt(os.path.join(output_dir, f"{turn}_pos.txt"), np.array([state.Bx, state.By, state.Wx, state.Wy], dtype=int))
        np.savetxt(os.path.join(output_dir, f"{turn}_row.txt"), np.array(state.row_wall, dtype=int))
        np.savetxt(os.path.join(output_dir, f"{turn}_column.txt"), np.array(state.column_wall, dtype=int))


def check_DP(feature_arr_list):
    wall0_count = 0
    jump_count = 0
    for feature_arr in feature_arr_list:
        is_wall0 = False
        is_jump = False

        prev_Bx = 4
        prev_By = 8
        prev_Wx = 4
        prev_Wy = 0
        for feature in feature_arr:
            Bx, By, Wx, Wy = get_pos(feature)
            Bwall = int(feature[0, 0, 2] * 10)
            Wwall = int(feature[0, 0, 3] * 10)
            if Bwall + Wwall == 0:
                is_wall0 = True
                d = abs(Bx - prev_Bx) + abs(By - prev_By) + abs(Wx - prev_Wx) + abs(Wy - prev_Wy)
                if d >= 2:
                    is_jump = True

            prev_Bx = Bx
            prev_By = By
            prev_Wx = Wx
            prev_Wy = Wy
        if is_wall0:
            wall0_count += 1
        if is_jump:
            jump_count += 1
            if jump_count == 14:
                print("="*50)
                print(jump_count)

                for turn, feature in enumerate(feature_arr):
                    display_state_from_feature(feature, turn, output_array=True)
                exit()
    print("wall0 count = {}/{}".format(wall0_count, len(feature_arr_list)))
    print("jump count = {}/{}".format(jump_count, len(feature_arr_list)))


def analyze_h5_main(args=None):
    np.seterr(divide='raise', invalid='raise')

    save_dir = os.path.join(TRAIN_LOG_DIR, "detail")
    os.makedirs(save_dir, exist_ok=True)

    use_load_df = os.path.exists(os.path.join(save_dir, "analyze_h5_result.csv"))

    if use_load_df:
        load_df = pd.read_csv(os.path.join(save_dir, "analyze_h5_result.csv"))
        load_div_epoch_list = list(load_df["div_epoch"])
        load_entropy_list = list(load_df["entropy"])
        load_win_v_post_mean_list = list(load_df["win_v_post_mean"])
        load_prev_post_diff_90_list = list(load_df["prev_post_diff_90"])
        load_prev_post_diff_95_list = list(load_df["prev_post_diff_95"])
        load_determine_wall_ratio_list = list(load_df["determine_wall_ratio"])
        load_placewall_ratio_list_dict = {}
        for key in PLACEWALL_DICT_KEYS:
            load_placewall_ratio_list_dict[key] = list(load_df["wall_ratio" + key])
        load_epoch_set = set([str(int(x)) for x in load_div_epoch_list])
    else:
        load_epoch_set = set([])

    #DATA_DIR = "backup/230117/train_results/data"

    each_data_dir_list = os.listdir(DATA_DIR)
    each_data_dir_list = sorted(each_data_dir_list, key=lambda s: int(s))[POOL_EPOCH_NUM:-1]
    #each_data_dir_list = each_data_dir_list[:140]

    # 計算済みのepochはあらかじめ計算対象から除外
    each_data_dir_list = list(set(each_data_dir_list) - load_epoch_set)
    each_data_dir_list = sorted(each_data_dir_list, key=lambda s: int(s))

    all_detail_epoch_list = []
    entropy_list = []
    win_v_post_mean_list = []
    prev_post_diff_90_list = []
    prev_post_diff_95_list = []
    determine_wall_ratio_list = []
    
    placewall_ratio_list_dict = {}
    for key in PLACEWALL_DICT_KEYS:
        placewall_ratio_list_dict[key] = []
    for each_data_dir in tqdm(each_data_dir_list):
        data_dir = os.path.join(DATA_DIR, each_data_dir)
        paths = os.listdir(data_dir)
        detail_epoch_list = [int(s.split(".")[0]) for s in paths if s.endswith(".h5")]
        detail_epoch_list = sorted(detail_epoch_list)
        all_detail_epoch_list.extend(detail_epoch_list)

        for div_i in range(len(detail_epoch_list) // LOAD_GAME_LIMIT):
            div_detail_epoch_list = detail_epoch_list[div_i * LOAD_GAME_LIMIT: (div_i + 1) * LOAD_GAME_LIMIT]

            epoch_h5arr_list_dict = {}
            for x_name in H5_NAME_LIST:
                epoch_h5arr_list_dict[x_name] = []

            for target_epoch in div_detail_epoch_list:
                # print("="*30)
                # print(target_epoch)
                h5file = h5py.File(os.path.join(data_dir, "{}.h5".format(target_epoch)), "r")

                for x_name in H5_NAME_LIST:
                    shape_dim = len(h5file[x_name].shape)
                    if shape_dim == 4:
                        arr = h5file[x_name][:, :, :, :]
                    elif shape_dim == 3:
                        arr = h5file[x_name][:, :, :]
                    elif shape_dim == 2:
                        arr = h5file[x_name][:, :]
                    elif shape_dim == 1:
                        arr = h5file[x_name][:]
                    epoch_h5arr_list_dict[x_name].append(arr)

            epoch_arr_dict = {}
            for x_name in H5_NAME_LIST:
                epoch_arr_dict[x_name] = np.concatenate(epoch_h5arr_list_dict[x_name], axis=0)

            # 棋譜を残すためにignore_augmented_data=Trueにしている
            arr_per_game_list_dict = divide_arr_to_games(epoch_arr_dict, ignore_augmented_data=True)

            placewall_dict, determine_wall_count = analyze_placewall(arr_per_game_list_dict)

            save_all_kifu(arr_per_game_list_dict, each_data_dir, div_i)

            entropy = calc_entropy(epoch_arr_dict["feature"])
            entropy_list.append(entropy)

            win_v_post = epoch_arr_dict["v_post"][epoch_arr_dict["reward"] == 1.0]
            win_v_post_mean_list.append(np.mean(win_v_post))

            placewall_num = sum(placewall_dict.values())
            for k, v in placewall_dict.items():
                placewall_ratio_list_dict[k].append(v / placewall_num)

            determine_wall_ratio_list.append(determine_wall_count / len(epoch_arr_dict["feature"]))

            prev_post_diff_90_list.append(np.percentile(np.abs(epoch_arr_dict["v_prev"] - epoch_arr_dict["v_post"]), 90))
            prev_post_diff_95_list.append(np.percentile(np.abs(epoch_arr_dict["v_prev"] - epoch_arr_dict["v_post"]), 95))

            # メモリ開放。これでも10GBは使う
            for x_name in H5_NAME_LIST:
                for arr in epoch_h5arr_list_dict[x_name]:
                    del arr
                del epoch_arr_dict[x_name]
            del arr_per_game_list_dict
            gc.collect()

    epoch_list = list(map(int, each_data_dir_list))
    div_epoch_list = []
    for epoch in epoch_list:
        div_epoch_list.extend([epoch + i / LOAD_GAME_LIMIT for i in range(LOAD_GAME_LIMIT)])

    if use_load_df:
        div_epoch_list = load_div_epoch_list + div_epoch_list
        entropy_list = load_entropy_list + entropy_list
        win_v_post_mean_list = load_win_v_post_mean_list + win_v_post_mean_list
        prev_post_diff_90_list = load_prev_post_diff_90_list + prev_post_diff_90_list
        prev_post_diff_95_list = load_prev_post_diff_95_list + prev_post_diff_95_list
        determine_wall_ratio_list = load_determine_wall_ratio_list + determine_wall_ratio_list
        for key in PLACEWALL_DICT_KEYS:
            placewall_ratio_list_dict[key] = load_placewall_ratio_list_dict[key] + placewall_ratio_list_dict[key]

    def calc_avg_list(x_list):
        ret = []
        for i in range(len(x_list) // LOAD_GAME_LIMIT):
            x_each = x_list[i * LOAD_GAME_LIMIT: (i + 1) * LOAD_GAME_LIMIT]
            ret.append(sum(x_each) / len(x_each))
        return ret
    
    div_avg_epoch_list = calc_avg_list(div_epoch_list)

    plt.clf()
    plt.plot(div_avg_epoch_list, calc_avg_list(entropy_list))
    plt.savefig(os.path.join(save_dir, "entropy.png"))

    plt.clf()
    plt.plot(div_avg_epoch_list, calc_avg_list(win_v_post_mean_list))
    plt.savefig(os.path.join(save_dir, "win_v_post_mean.png"))

    plt.clf()
    plt.plot(div_avg_epoch_list, calc_avg_list(prev_post_diff_90_list), label="abs(prev - post)  90% percentile")
    plt.plot(div_avg_epoch_list, calc_avg_list(prev_post_diff_95_list), label="abs(prev - post)  95% percentile")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "prev_post_v_diff.png"))

    plt.clf()
    for key in PLACEWALL_DICT_KEYS:
        plt.plot(div_avg_epoch_list, calc_avg_list(placewall_ratio_list_dict[key]), label=key)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "wall_ratio.png"))

    plt.clf()
    plt.plot(div_avg_epoch_list, calc_avg_list(determine_wall_ratio_list), label="determine_wall_ratio")
    plt.savefig(os.path.join(save_dir, "determine_wall_ratio.png"))

    # 計算結果を保存しておいて再利用できるようにする
    save_df = pd.DataFrame({
        "div_epoch": div_epoch_list,
        "entropy": entropy_list,
        "win_v_post_mean": win_v_post_mean_list,
        "prev_post_diff_90": prev_post_diff_90_list,
        "prev_post_diff_95": prev_post_diff_95_list,
        "determine_wall_ratio": determine_wall_ratio_list
    })
    for key in PLACEWALL_DICT_KEYS:
        save_df["wall_ratio" + key] = placewall_ratio_list_dict[key]

    save_df.to_csv(os.path.join(save_dir, "analyze_h5_result.csv"))
    

def watch_selfplay_games():
    
    each_data_dir_list = os.listdir(DATA_DIR)
    each_data_dir_list = sorted(each_data_dir_list, key=lambda s: int(s))
    print("candidate: {}~{}".format(each_data_dir_list[0], each_data_dir_list[-2]))
    print("input target epoch")
    each_data_dir = input()
    print("input ignore_augmented_data (True, False)")
    ignore_augmented_data = (input() == "True")
    all_detail_epoch_list = []

    data_dir = os.path.join(DATA_DIR, each_data_dir)
    paths = os.listdir(data_dir)
    detail_epoch_list = [int(s.split(".")[0]) for s in paths if s.endswith(".h5")]
    detail_epoch_list = sorted(detail_epoch_list)
    detail_epoch_list = detail_epoch_list[:LOAD_GAME_LIMIT]  # メモリ使い果たすの対策
    all_detail_epoch_list.extend(detail_epoch_list)

    epoch_h5arr_list_dict = {}
    for x_name in H5_NAME_LIST:
        epoch_h5arr_list_dict[x_name] = []

    for target_epoch in detail_epoch_list:
        # print("="*30)
        # print(target_epoch)
        h5file = h5py.File(os.path.join(data_dir, "{}.h5".format(target_epoch)), "r")

        for x_name in H5_NAME_LIST:
            shape_dim = len(h5file[x_name].shape)
            if shape_dim == 4:
                arr = h5file[x_name][:, :, :, :]
            elif shape_dim == 3:
                arr = h5file[x_name][:, :, :]
            elif shape_dim == 2:
                arr = h5file[x_name][:, :]
            elif shape_dim == 1:
                arr = h5file[x_name][:]
            epoch_h5arr_list_dict[x_name].append(arr)

    epoch_arr_dict = {}
    for x_name in H5_NAME_LIST:
        epoch_arr_dict[x_name] = np.concatenate(epoch_h5arr_list_dict[x_name], axis=0)
        # for x in epoch_h5arr_list_dict[x_name]:
        #     del x

    arr_per_game_list_dict = divide_arr_to_games(epoch_arr_dict, ignore_augmented_data)
    
    for i, feature_arr in enumerate(arr_per_game_list_dict["feature"]):
        for turn, feature in enumerate(feature_arr):
            display_state_from_feature(feature, turn)
            print("pi =")
            display_parameter(np.asarray(arr_per_game_list_dict["pi"][i][turn] * 1000, dtype="int32"))
            print("next_pi =")
            display_parameter(np.asarray(arr_per_game_list_dict["next_pi"][i][turn] * 1000, dtype="int32"))
            print("reward =", arr_per_game_list_dict["reward"][i][turn])
            print("dist_diff =", arr_per_game_list_dict["dist_diff"][i][turn])
            print("black_walls =", arr_per_game_list_dict["black_walls"][i][turn])
            print("white_walls =", arr_per_game_list_dict["white_walls"][i][turn])
            print("remaining_turn_num =", arr_per_game_list_dict["remaining_turn_num"][i][turn])
            print("remaining_black_moves =", arr_per_game_list_dict["remaining_black_moves"][i][turn])
            print("remaining_white_moves =", arr_per_game_list_dict["remaining_white_moves"][i][turn])
            print("row_wall.T =")
            print(arr_per_game_list_dict["row_wall"][i][turn].T)
            print("column_wall.T =")
            print(arr_per_game_list_dict["column_wall"][i][turn].T)
            print("dist_array1.T =")
            print(arr_per_game_list_dict["dist_array1"][i][turn].T)
            print("dist_array2.T =")
            print(arr_per_game_list_dict["dist_array2"][i][turn].T)
            print("B_traversed_arr.T =")
            print(arr_per_game_list_dict["B_traversed_arr"][i][turn].T)
            print("W_traversed_arr.T =")
            print(arr_per_game_list_dict["W_traversed_arr"][i][turn].T)
        print()
        print(f"↑ game {i} ↑")
        print("~"*30)
        print("press any key")
        input()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="analyze_h5")
    args = parser.parse_args()
    if args.mode == "analyze_h5":
        analyze_h5_main()
    elif args.mode == "watch_selfplay_games":
        watch_selfplay_games()
