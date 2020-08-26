# coding:utf-8
from Agent import actionid2str
from State import State, CHANNEL
from State import DRAW_TURN
from Human import Human
from LinearAI import LinearAI
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 学習設定
MAX_DATA = 600000
STEP = 20000
TRAIN_CYCLE = 10  # 何epochおきに学習・評価を行うか
LEARN_REP_NUM = MAX_DATA // STEP * 3
MAX_EPOCH = 1000
TAU_MIN = 0.25
MAX_AI_NUM_FOR_EVALUATE = 3
EVALUATE_PLAY_NUM = 150
RATE_TH = 0.1  # レートがいくつ以上あがったら新AIとして採用するか
RATE_TH2 = 0.5  # レートがいくつ以上あがったらAI listを更新するか


# agentsは初期化されてるとする
def normal_play(agents):
    state = State()
    while True:
        state.display_cui()
        start = time.time()
        s = agents[0].act(state, showNQ=True)
        end = time.time()
        if isinstance(s, int):
            a = actionid2str(state, s)
        else:
            a = s
        while not state.accept_action_str(a):
            print(a)
            print("this action is impossible")
            s = agents[0].act(state, showNQ=True)
            if isinstance(s, int):
                a = actionid2str(state, s)
            else:
                a = s
        agents[1].prev_action = s

        if state.terminate:
            break
        #time.sleep(0.1)

        state.display_cui()
        s = agents[1].act(state, showNQ=True)
        if isinstance(s, int):
            a = actionid2str(state, s)
        else:
            a = s
        while not state.accept_action_str(a):
            print(a)
            print("this action is impossible")
            s = agents[1].act(state, showNQ=True)
            if isinstance(s, int):
                a = actionid2str(state, s)
            else:
                a = s
        agents[0].prev_action = s

        #time.sleep(0.1)
        if state.terminate:
            break

    state.display_cui()
    print("The game finished. reward=({}, {})".format(state.reward, -state.reward))


def generate_data(AIs, play_num, noise=0.1, display=False, equal_draw=True, info=False):
    data = []
    hash_ = 0
    for j in range(play_num):
        state = State()
        AIs[0].init_prev()
        AIs[1].init_prev()
        featuress = [[], [], [], []]
        for i, b1, b2 in [(0, False, False), (1, True, False), (2, False, True), (3, True, True)]:
            featuress[i].append(state.feature_CNN(b1, b2))

        pis = []
        states = [state_copy(state)]
        while True:
            AIs[0].tau = np.random.rand() * (1. - TAU_MIN) + TAU_MIN
            AIs[1].tau = np.random.rand() * (1. - TAU_MIN) + TAU_MIN
            if state.turn >= 20:
                AIs[0].tau = TAU_MIN
                AIs[1].tau = TAU_MIN
            s, pi = AIs[0].act_and_get_pi(state, noise=noise)
            a = actionid2str(state, s)
            while not state.accept_action_str(a):
                print("this action is impossible")
                print(a)
                s.display_cui()
                exit()
            AIs[1].prev_action = s
            pis.append(pi)
            if display:
                state.display_cui()
            state.check_placable_array_algo()
            end = False
            for state2 in states:
                if equal_draw and state == state2:
                    end = True
                    break
            if end:
                break
            states.append(state_copy(state))
            if state.terminate:
                break
            for i, b1, b2 in [(0, False, False), (1, True, False), (2, False, True), (3, True, True)]:
                featuress[i].append(state.feature_CNN(b1, b2))
            s, pi = AIs[1].act_and_get_pi(state, noise=noise)
            a = actionid2str(state, s)
            while not state.accept_action_str(a):
                print("this action is impossible")
                print(a)
                s.display_cui()
                exit()
            AIs[0].prev_action = s
            pis.append(pi)
            if display:
                state.display_cui()
            state.check_placable_array_algo()
            end = False
            for state2 in states:
                if equal_draw and state == state2:
                    end = True
                    break
            if end:
                break
            states.append(state_copy(state))
            if state.terminate:
                break
            for i, b1, b2 in [(0, False, False), (1, True, False), (2, False, True), (3, True, True)]:
                featuress[i].append(state.feature_CNN(b1, b2))
        del states
        hash_ += state.turn
        if state.reward == 0:
            continue
        for feature1, feature2, feature3, feature4, pi in zip(featuress[0], featuress[1], featuress[2], featuress[3], pis):
            data.append((feature1, pi, state.reward))
            a = np.flip(pi[:64].reshape((8, 8)), 0).flatten()
            b = np.flip(pi[64:128].reshape((8, 8)), 0).flatten()
            mvarray1 = pi[128:].reshape((3, 3))
            mvarray2 = np.zeros((3, 3))
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    mvarray2[x, y] = mvarray1[-x, y]
            c = mvarray2.flatten()
            data.append((feature2, np.concatenate([a, b, c]), state.reward))
            a = np.flip(pi[:64].reshape((8, 8)), 1).flatten()
            b = np.flip(pi[64:128].reshape((8, 8)), 1).flatten()
            mvarray1 = pi[128:].reshape((3, 3))
            mvarray2 = np.zeros((3, 3))
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    mvarray2[x, y] = mvarray1[x, -y]
            c = mvarray2.flatten()
            data.append((feature3, np.concatenate([a, b, c]), -state.reward))
            a = np.flip(np.flip(pi[:64].reshape((8, 8)), 1), 0).flatten()
            b = np.flip(np.flip(pi[64:128].reshape((8, 8)), 1), 0).flatten()
            mvarray1 = pi[128:].reshape((3, 3))
            mvarray2 = np.zeros((3, 3))
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    mvarray2[x, y] = mvarray1[-x, -y]
            c = mvarray2.flatten()
            data.append((feature4, np.concatenate([a, b, c]), -state.reward))
    if info:
        print("hash = {}".format(hash_))

    return data


# 中で先手後手を順番に入れ替えている
def evaluate(AIs, play_num, return_draw=False):
    wins = 0.
    draw_num = 0
    for i in range(play_num):
        state = State()
        AIs[0].init_prev()
        AIs[1].init_prev()
        AIs[i % 2].color = 0
        AIs[1 - i % 2].color = 1
        while True:
            s, pi = AIs[i % 2].act_and_get_pi(state)
            a = actionid2str(state, s)
            while not state.accept_action_str(a):
                print("this action is impossible")
                s, pi = AIs[i % 2].act_and_get_pi(state)
                a = actionid2str(state, s)
            AIs[1 - i % 2].prev_action = s

            if state.terminate:
                break

            s, pi = AIs[1 - i % 2].act_and_get_pi(state)
            a = actionid2str(state, s)
            while not state.accept_action_str(a):
                print("this action is impossible")
                s, pi = AIs[1 - i % 2].act_and_get_pi(state)
                a = actionid2str(state, s)
            AIs[i % 2].prev_action = s

            if state.terminate:
                break

        if i % 2 == 0 and state.reward == 1:
            wins += 1.
        elif i % 2 == 1 and state.reward == -1:
            wins += 1.
        elif state.reward == 0:
            wins += 0.5
            draw_num += 1
        sys.stderr.write('\r\033[K {}win/{}'.format(i + 1 - wins, i + 1))
        sys.stderr.flush()
    print("")
    AIs[0].color = 0
    AIs[1].color = 1

    if return_draw:
        return wins, draw_num
    else:
        return wins


def debug_learn(AIs):
    h5_num = MAX_DATA // STEP
    AIs[1].learn(h5_num - 1, h5_num, LEARN_REP_NUM, STEP)
    # ----------evaluation--------
    AIs[0].tau = TAU_MIN
    AIs[1].tau = TAU_MIN
    AIs[0].search_nodes = search_nodes
    AIs[1].search_nodes = search_nodes

    play_num = 100
    white_win_num = evaluate(AIs, play_num)
    win_rate = (play_num - white_win_num) / play_num
    print("new AI win rate={}".format(win_rate))
    AIs[1].save("./parameter/debug.ckpt")


def learn(AIs, restart=False, initial_rate=0., skip_first_selfplay=False, restart_filename="restart.pkl"):
    features = np.zeros((STEP, 9, 9, CHANNEL))
    pis = np.zeros((STEP, 137))
    rewards = np.zeros((STEP,))

    AI_id_list = [-1]  # レート測定のためのAIの番号リスト。-1は初期AIを表す。
    AI_rate_list = [initial_rate]
    AIs[1].save("./parameter/init.ckpt")

    search_nodes = AIs[0].search_nodes
    initial_epoch = 0
    prev_rate = initial_rate  # 学習に使っているAIのレート
    if restart:
        initial_epoch, AI_id_list, AI_rate_list, prev_rate = pickle.load(open(restart_filename, "rb"))
        print("retrain: {}, {}, {}, {}".format(initial_epoch, AI_id_list, AI_rate_list, prev_rate))
        if not skip_first_selfplay:
            initial_epoch += 1
        AI_id = AI_id_list[-1]
        if AI_id == -1:
            AIs[0].load("./parameter/init.ckpt")
            AIs[1].load("./parameter/init.ckpt")
            AIs[0].v_is_dist = True
            AIs[1].v_is_dist = True
        else:
            AIs[0].load("./parameter/epoch{}.ckpt".format(AI_id))
            AIs[1].load("./parameter/epoch{}.ckpt".format(AI_id))

    h5_num = MAX_DATA // STEP
    for i in range(initial_epoch, MAX_EPOCH):
        # -------data generation-------
        b_win = 0
        w_win = 0
        draw = 0
        index = 0
        if not skip_first_selfplay:
            start = time.time()
            while index < STEP:
                if i < h5_num:
                    temp_data = generate_data(AIs, 1, noise=1., equal_draw=False)
                else:
                    temp_data = generate_data(AIs, 1, noise=0.1)
                if len(temp_data) == 0:
                    draw += 1
                else:
                    reward = -temp_data[-1][2]  # augmentationの影響で報酬がひっくり返っている
                    if reward == 1:
                        b_win += 1
                    elif reward == -1:
                        w_win += 1
                for feature, pi, reward in temp_data:
                    features[index] = feature
                    pis[index] = pi
                    rewards[index] = reward
                    index += 1
                    if index >= STEP:
                        break
                del temp_data
                sys.stderr.write('\r\033[Kepoch{}:data={}/{} B{}win W{}win {}draw'.format(i + 1, index, STEP, b_win, w_win, draw))
                sys.stderr.flush()
            h5file = h5py.File("data/{}.h5".format(i), "w")
            h5file.create_dataset("feature", data=features, compression="gzip", compression_opts=1)
            h5file.create_dataset("pi", data=pis, compression="gzip", compression_opts=1)
            h5file.create_dataset("reward", data=rewards, compression="gzip", compression_opts=1)
            h5file.flush()
            h5file.close()

            print("")
            print("elapsed time(self-play)={}".format(time.time() - start))
            pickle.dump((i, AI_id_list, AI_rate_list, prev_rate), open(restart_filename, "wb"))
        skip_first_selfplay = False
        start = time.time()

        if i < h5_num - 1:
            continue
        if i % TRAIN_CYCLE != TRAIN_CYCLE - 1:
            continue

        # --------training---------
        # 全体を通しAI[1]が新しいAIとする
        AIs[1].save("./parameter/pre.ckpt")
        if i >= h5_num:
            AIs[1].load("./parameter/post.ckpt")
        AIs[1].learn(i, h5_num, LEARN_REP_NUM, STEP)
        AIs[1].save("./parameter/epoch{}.ckpt".format(i + 1))

        print("elapsed time(training)={}".format(time.time() - start))
        start = time.time()

        # ----------evaluation--------
        AIs[0].tau = TAU_MIN
        AIs[1].tau = TAU_MIN
        AIs[0].search_nodes = search_nodes
        AIs[1].search_nodes = search_nodes
        AIs[0].v_is_dist = False
        AIs[1].v_is_dist = False
        win_num_list = []
        play_num = EVALUATE_PLAY_NUM // len(AI_id_list)
        for id_, old_rate in zip(AI_id_list, AI_rate_list):
            if id_ == -1:
                AIs[0].load("./parameter/init.ckpt")
                AIs[0].v_is_dist = True
                AIs[0].p_is_almost_flat = True
            else:
                AIs[0].load("./parameter/epoch{}.ckpt".format(id_))
                AIs[0].v_is_dist = False
                AIs[0].p_is_almost_flat = False
            new_ai_win_num = play_num - evaluate(AIs, play_num)
            win_num_list.append(new_ai_win_num)
            print("new AI vs {} (rate={:.3f}) : {}/{}, win rate={:.3f}".format(id_, old_rate, new_ai_win_num, play_num, new_ai_win_num / play_num))
        AIs[0].p_is_almost_flat = False
        new_rate = calc_rate(play_num, np.array(AI_rate_list), np.array(win_num_list))

        AIs[1].save("./parameter/post.ckpt")
        if new_rate >= prev_rate + RATE_TH:
            prev_rate = new_rate
            print("new AI accepted. rate={}".format(new_rate))
            AIs[0].load("./parameter/post.ckpt")

            if new_rate >= AI_rate_list[-1] + RATE_TH2:
                if len(AI_id_list) >= MAX_AI_NUM_FOR_EVALUATE:
                    AI_id_list.pop(0)
                    AI_rate_list.pop(0)
                AI_id_list.append(i + 1)
                AI_rate_list.append(new_rate)
                print(AI_id_list, AI_rate_list)

            # どちらもNNでvを計算するようにする
            AIs[0].v_is_dist = False
        else:
            print("new AI rejected")
            AIs[1].load("./parameter/pre.ckpt")

        gc.collect()
        print("elapsed time(evaluation)={}".format(time.time() - start))
        print("")

        pickle.dump((i, AI_id_list, AI_rate_list, prev_rate), open(restart_filename, "wb"))
        with open("train_log.csv", "a") as file:
            file.write("{},{},{},{},{},{},{}".format(datetime.datetime.today(),
                                            check_output(["git", "show", '--format="%H"', "-s"]).rstrip(),
                                            i, prev_rate, b_win, w_win, draw) + os.linesep)

        # 元に戻す
        AIs[0].tau = 1.
        AIs[1].tau = 1.
        AIs[0].search_nodes = search_nodes
        AIs[1].search_nodes = search_nodes


if __name__ == '__main__':
    np.seterr(divide='raise', invalid='raise')
    search_nodes = 300
    if len(sys.argv) >= 3:
        search_nodes = int(sys.argv[2])

    if sys.argv[1] == "train":
        # 1つ目はパラメータ0にすることでランダムAIにする.2つ目は学習のことがあるから0にしてはだめ
        # 200208現在v固定中
        learn([CNNAI(0, search_nodes=search_nodes, all_parameter_zero=True, v_is_dist=True),
               CNNAI(1, search_nodes=search_nodes, v_is_dist=True)])
    elif sys.argv[1] == "retrain":
        with open("train_log.csv", "r") as f:
            s = f.read().strip().split("\n")[-1]
        slist = s.split(",")
        rate = float(slist[3])
        learn([CNNAI(0, search_nodes=search_nodes, all_parameter_zero=True),
               CNNAI(1, search_nodes=search_nodes)],
              restart=True, initial_rate=rate, skip_first_selfplay=False)
    elif sys.argv[1] == "debug_learn":
        # １つ目のAIを距離だけで考えるものとして、そこからどれだけ学習の結果強くなれるかを検証する
        debug_learn([CNNAI(0, search_nodes=search_nodes, v_is_dist=True, p_is_almost_flat=True), CNNAI(1, search_nodes=search_nodes, v_is_dist=True)])
    elif sys.argv[1] == "view":
        AIs = [CNNAI(0, search_nodes=search_nodes, tau=0.5), CNNAI(1, search_nodes=search_nodes, tau=0.5)]
        AIs[0].load("./parameter/post.ckpt")
        AIs[1].load("./parameter/post.ckpt")
        normal_play(AIs)
    elif sys.argv[1] == "vs":
        AIs = [Human(0), CNNAI(1, search_nodes=search_nodes, tau=0.5)]
        AIs[1].load("./parameter/post.ckpt")
        normal_play(AIs)
    elif sys.argv[1] == "test":
        #np.random.seed(0)
        AIs = [CNNAI(0, search_nodes=search_nodes, tau=0.5), CNNAI(1, search_nodes=search_nodes, tau=0.5)]
        AIs[0].load("./parameter/epoch8.ckpt")
        AIs[1].load("./parameter/epoch8.ckpt")
        for i in range(100):
            print("============={}==============".format(i))
            normal_play(AIs)
    elif sys.argv[1] == "measure":
        np.random.seed(0)
        game_num = 10
        AIs = [CNNAI(0, search_nodes=search_nodes, tau=1), CNNAI(1, search_nodes=search_nodes, tau=1)]
        AIs[0].load("./parameter/post.ckpt")
        AIs[1].load("./parameter/post.ckpt")

        start_time = time.time()
        temp_data = generate_data(AIs, game_num, noise=0.1, display=True, info=True)
        with open("measure.log", "a") as f:
            f.write("{},{},{},{}".format(datetime.datetime.today(),
                                      check_output(["git", "show", '--format="%H"', "-s"]).rstrip(),
                                      time.time() - start_time, game_num) + os.linesep)
