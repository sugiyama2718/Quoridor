# coding:utf-8
from Agent import actionid2str
from State import State
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

# 学習設定
MAX_DATA = 2000000
STEP = 250000
SEARCH_NODE = 300
MAX_EPOCH = 100
TAU_MIN = 0.25


# agentsは初期化されてるとする
def normal_play(agents):
    state = State()
    while True:
        state.display_cui()
        start = time.time()
        s = agents[0].act(state, showNQ=True)
        end = time.time()
        print(end - start)
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


def generate_data(AIs, play_num, noise=0.1, display=False, equal_draw=True):
    data = []
    for i in range(play_num):
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
                s, pi = AIs[0].act_and_get_pi(state)
                a = actionid2str(state, s)
            AIs[1].prev_action = s
            pis.append(pi)
            if display:
                state.display_cui()
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
                s, pi = AIs[1].act_and_get_pi(state)
                a = actionid2str(state, s)
            AIs[0].prev_action = s
            pis.append(pi)
            if display:
                state.display_cui()
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


def learn(AIs, restart=False, initial_rate=0., initial_epoch=0, skip_first_selfplay=False):
    features = np.zeros((STEP, 8, 8, 7))
    pis = np.zeros((STEP, 137))
    rewards = np.zeros((STEP,))

    search_nodes = AIs[0].search_nodes
    if restart:
        if os.path.exists("./parameter/post.ckpt"):
            AIs[0].load("./parameter/post.ckpt")
            AIs[1].load("./parameter/post.ckpt")
    else:
        for filename in os.listdir("data"):
            os.remove("data/" + filename)
        # 初回のみ探索ノード1(引き分け多いから)

        AIs[0].search_nodes = 1
        AIs[1].search_nodes = 1

    rating = initial_rate
    h5_num = MAX_DATA // STEP
    for i in range(initial_epoch, MAX_EPOCH):
        if not skip_first_selfplay:
            # -------data generation-------
            b_win = 0
            w_win = 0
            draw = 0
            start = time.time()
            index = 0
            while index < STEP:
                if i < h5_num and not restart:
                    #State.draw_turn = 200
                    temp_data = generate_data(AIs, 1, noise=1., equal_draw=False)
                    #State.draw_turn = DRAW_TURN
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
        skip_first_selfplay = False
        start = time.time()

        if i < h5_num - 1:
            continue
        elif i >= h5_num:
            pass
            #os.remove("data/{}.h5".format(i - h5_num))

        # --------training---------
        AIs[1].save("./parameter/pre.ckpt")
        if i >= h5_num:
            AIs[1].load("./parameter/post.ckpt")
        AIs[1].learn(i, h5_num, h5_num, STEP)
        AIs[1].save("./parameter/post.ckpt")

        print("elapsed time(training)={}".format(time.time() - start))
        start = time.time()

        # ----------evaluation--------
        AIs[0].tau = TAU_MIN
        AIs[1].tau = TAU_MIN
        AIs[0].search_nodes = search_nodes
        AIs[1].search_nodes = search_nodes
        play_num = 100
        white_win_num = evaluate(AIs, play_num)
        win_rate = (play_num - white_win_num) / play_num
        print("new AI win rate={}".format(win_rate))
        if win_rate > 0.5:
            win_rate2 = (play_num + 2 - (white_win_num + 1)) / (play_num + 2)
            rating += -np.log(1. / win_rate2 - 1)
            print("new AI accepted. rate={}".format(rating))
            AIs[1].save("./parameter/epoch{}.ckpt".format(i + 1))
            AIs[0].load("./parameter/post.ckpt")
        else:
            print("new AI rejected")
            AIs[1].load("./parameter/pre.ckpt")

        gc.collect()
        print("elapsed time(evaluation)={}".format(time.time() - start))
        print("")

        # 元に戻す
        AIs[0].tau = 1.
        AIs[1].tau = 1.
        AIs[0].search_nodes = search_nodes
        AIs[1].search_nodes = search_nodes

search_nodes = 300
if len(sys.argv) >= 3:
    search_nodes = int(sys.argv[2])

if sys.argv[1] == "train":
    # 1つ目はパラメータ0にすることでランダムAIにする.2つ目は学習のことがあるから0にしてはだめ
    learn([CNNAI(0, search_nodes=SEARCH_NODE, all_parameter_zero=True),
           CNNAI(1, search_nodes=SEARCH_NODE)])
          #restart=True, initial_rate=7, initial_epoch=17, skip_first_selfplay=False)
elif sys.argv[1] == "view":
    AIs = [CNNAI(0, search_nodes=search_nodes, tau=0.25), CNNAI(1, search_nodes=search_nodes, tau=0.25)]
    AIs[0].load("./parameter/post.ckpt")
    AIs[1].load("./parameter/post.ckpt")
    normal_play(AIs)
elif sys.argv[1] == "measure":
    np.random.seed(0)
    game_num = 20
    AIs = [CNNAI(0, search_nodes=search_nodes, tau=1), CNNAI(1, search_nodes=search_nodes, tau=1)]
    AIs[0].load("./parameter/epoch22.ckpt")
    AIs[1].load("./parameter/epoch22.ckpt")
    temp_data = generate_data(AIs, game_num, noise=0.1, display=True)

"""
print("---------")
for node_num in [1, 10, 25, 50, 100]:
    AIs[0].search_nodes = node_num
    AIs[1].search_nodes = node_num
    win_num, draw = evaluate(AIs, game_num, True)
    print("{}win {}draw / {}games".format(win_num, draw, game_num))

print("---------")
AIs[1].search_nodes = 1
for node_num in [1, 10, 25, 50, 100]:
    AIs[0].search_nodes = node_num
    win_num, draw = evaluate(AIs, game_num, True)
    print("{}win {}draw / {} games".format(win_num, draw, game_num))

AIs = [CNNAI(0, search_nodes=50, tau=0.5), CNNAI(1, search_nodes=50, tau=0.5)]
AIs[0].load("./parameter/post.ckpt")
AIs[1].load("./parameter/post.ckpt")

print("---------")
for C_puct in [3, 4, 5]:
    AIs[0].C_puct = C_puct
    win_num, draw = evaluate(AIs, game_num, True)
    print("{}win {}draw / {} games".format(win_num, draw, game_num))

print("---------")
AIs[0].C_puct = 2
for tau in [0.25, 0]:
    AIs[0].tau = tau
    win_num, draw = evaluate(AIs, game_num, True)
    print("{}win {}draw / {} games".format(win_num, draw, game_num))

print("---------")
AIs[0].tau = 0.5
AIs[0].n_parallel = 4
for node_num in [1, 10, 25, 50, 100]:
    AIs[0].search_nodes = node_num
    AIs[1].search_nodes = node_num
    win_num, draw = evaluate(AIs, game_num, True)
    print("{}win {}draw / {}games".format(win_num, draw, game_num))
"""

