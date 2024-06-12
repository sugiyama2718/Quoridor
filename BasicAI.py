# coding:utf-8
#from memory_profiler import profile
from Agent import Agent, actionid2str, move_id2dxdy, is_jump_move, dxdy2actionid, str2actionid
from Tree import Tree, Tree_c
import State
from State import State, State_init, color_p, movable_array, movable_array_flatten, accept_action_str, BOARD_LEN, get_player_dist_from_goal, is_certain_path_terminate, placable_flatten_array, calc_dist_array, display_cui, feature_int, get_arrays_for_feature_CNN
import numpy as np
import copy
from graphviz import Digraph
import os
from itertools import product
import time
from pprint import pprint
import random
from config import N_PARALLEL, SHORTEST_N_RATIO, SHORTEST_Q
from config import *
from util import Glendenning2Official, Official2Glendenning
import ctypes

num2str = {0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f", 6:"g", 7:"h", 8:"i"}

if os.name == "nt":
    lib = ctypes.CDLL('./State_util.dll')
else:
    lib = ctypes.CDLL('./State_util.so')

select_action = lib.select_action
select_action.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
                              ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int)
select_action.restype = ctypes.c_int

copy_state_c = lib.copy_state
copy_state_c.argtypes = [ctypes.POINTER(State), ctypes.POINTER(State)]
copy_state_c.restype = None

copy_int_arr = lib.copyIntArr
copy_int_arr.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
copy_int_arr.restype = None

copy_float_arr = lib.copyFloatArr
copy_float_arr.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float]
copy_float_arr.restype = None

mult_int_arr = lib.multIntArr
mult_int_arr.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
mult_int_arr.restype = None

mult_float_arr = lib.multFloatArr
mult_float_arr.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
mult_float_arr.restype = None

add_virtual_loss = lib.add_virtual_loss
add_virtual_loss.argtypes = [ctypes.POINTER(Tree_c), ctypes.c_int, ctypes.c_int, ctypes.c_int]
add_virtual_loss.restype = None


def get_state_vec(state):
    # stateを固定長タプルにしてdictのkeyにするために使う。state.turnを入れているのは、turnの異なる状態を区別して無限ループを避けるため
    return tuple([state.turn] + list(feature_int(state).flatten()))


def get_state_vec_from_tree(tree):
    # state_vecをまだ計算していないときだけ計算して格納することで高速化
    if tree.state_vec is None:
        tree.state_vec = get_state_vec(tree.s)
    return tree.state_vec


def display_parameter(x):
    a = x[:64].reshape((8, 8))
    b = x[64:128].reshape((8, 8))
    c = x[128:].reshape((3, 3))
    for y in range(8):
        for x in range(8):
            print("{:5}".format(a[x, y]), end="")
        print("  ", end="")
        for x in range(8):
            print("{:5}".format(b[x, y]), end="")
        print("")
    for y in [-1, 0, 1]:
        for x in [-1, 0, 1]:
            print("{:5}".format(c[x, y]), end="")
        print("")


# 引数のgにgraphviz用のグラフを入れる。ノード共有のある木構造向け。
node_id = 0
def get_graphviz_tree_for_shared_tree(tree, g, threshold=5, root=True, color=None, visited=None):
    global node_id
    if tree is None:
        return
    if root:
        color = tree.s.turn % 2
        init_visited = set([])
        def init_tree(t):
            t.node_id = None
            state_vec = get_state_vec(t.s)
            if state_vec in init_visited:
                return 
            
            init_visited.add(state_vec)
            for key, child_tree in t.children.items():
                init_tree(child_tree)
        init_tree(tree)
        visited = set([])
        node_id = 0
    if sum(tree.tree_c.contents.N_arr) == 0:
        # g.node(str(node_id), label="end")
        # node_id += 1
        return
    
    g.node(str(tree.node_id), label=str(int(sum(tree.tree_c.contents.N_arr))) + os.linesep + "{:.3f}".format(sum(tree.tree_c.contents.W_arr) / sum(tree.tree_c.contents.N_arr)))

    max_N = int(max(tree.tree_c.contents.N_arr))
    for key in range(137):
        if int(tree.tree_c.contents.N_arr[key]) == 0:
            continue

        penwidth = "1"
        if int(tree.tree_c.contents.N_arr[key]) == max_N:
            penwidth = "3"

        if key in tree.children.keys():
            value = tree.children[key]
            if int(tree.tree_c.contents.N_arr[key]) >= threshold:
                state_vec = get_state_vec(value.s)
                if state_vec in visited:
                    # 探索済みの場合、既にnode_idが付与済みなのでそれに接続
                    g.edge(str(tree.node_id), str(value.node_id), label=Glendenning2Official(actionid2str(tree.s, key)) + os.linesep + str(int(tree.tree_c.contents.N_arr[key])) + os.linesep + "{:.1f}%".format(100 * tree.P[key]), penwidth=penwidth)
                    continue

                visited.add(state_vec)
                value.node_id = node_id
                node_id += 1
                g.edge(str(tree.node_id), str(value.node_id), label=Glendenning2Official(actionid2str(tree.s, key)) + os.linesep + str(int(tree.tree_c.contents.N_arr[key])) + os.linesep + "{:.1f}%".format(100 * tree.P[key]), penwidth=penwidth)

                get_graphviz_tree_for_shared_tree(value, g, threshold=threshold, root=False, color=color, visited=visited)

        else:
            if int(tree.tree_c.contents.N_arr[key]) >= threshold:
                g.edge(str(tree.node_id), str(node_id), label=Glendenning2Official(actionid2str(tree.s, key)) + os.linesep + str(int(tree.tree_c.contents.N_arr[key])) + os.linesep + "{:.1f}".format(100 * tree.P[key]), penwidth=penwidth)
                state = state_copy(tree.s)
                accept_action_str(state, actionid2str(tree.s, key))
                if color == 0:
                    v = state.pseudo_reward
                else:
                    v = -state.pseudo_reward
                g.node(str(node_id), label="end" + os.linesep + str(v))

                node_id += 1


def state_copy(s):
    ret = State()
    copy_state_c(ret, s)
    return ret


def calc_optimal_move_by_DP(s):
    s = state_copy(s)
    boards = list(product(range(9), range(9), range(9), range(9), range(2)))  # 壁0なので、(Bx, By, Wx, Wy, 手番)で盤面が一意に定まる。のでそれで盤面を表現。

    dist_array1 = calc_dist_array(s, 0)
    dist_array2 = calc_dist_array(s, BOARD_LEN - 1)

    # 合法手のみに絞る
    def is_legal(board):
        Bx, By, Wx, Wy, _ = board
        if Bx == Wx and By == Wy:
            return False
        Bd = dist_array1[Bx, By]
        Wd = dist_array2[Wx, Wy]
        if Bd == 0 and Wd == 0:
            return False

        for is_black in [False, True]:
            if is_black:
                x = Bx
                y = By
                s.turn = 0
            else:
                x = Wx
                y = Wy
                s.turn = 1
            movable_arr = movable_array(s, x, y, shortest_only=False)
            if not np.any(movable_arr):
                return False

        return True
    boards = list(filter(is_legal, boards))

    # ゴールからの距離を組にする
    boards_d = [(-1, -1, -1, -1, -1, -1, -1)] * len(boards)
    for i, board in enumerate(boards):
        Bx, By, Wx, Wy, is_black = board
        Bd = dist_array1[Bx, By]
        Wd = dist_array2[Wx, Wy]
        boards_d[i] = (Bx, By, Wx, Wy, is_black, Bd, Wd)

    # ゴールからの距離の和でソート
    boards_d = sorted(boards_d, key=lambda x: x[5] + x[6])

    optimal_dict = {}
    for Bx, By, Wx, Wy, is_black, Bd, Wd in boards_d:
        if Bd == 0 or Wd == 0:
            optimal_dict[(Bx, By, Wx, Wy, is_black)] = (-1, -1, -1, -1, is_black, Wd - Bd, Bd != Wd)  # 次のboardと、B目線でどれだけ差をつけて勝ったか？Bから最大化したい量、有効な要素かどうか
            continue

        if is_black:
            x = Bx
            y = By
            d = Bd
            s.turn = 0
        else:
            x = Wx
            y = Wy
            d = Wd
            s.turn = 1
        s.Bx = Bx
        s.By = By
        s.Wx = Wx
        s.Wy = Wy
        movable_arr = movable_array(s, x, y, shortest_only=True)
        if not np.any(movable_arr):
            movable_arr = movable_array(s, x, y, shortest_only=False)

        # 距離を縮める方向に動けているなら探索済みでoptimal_dictに要素が存在するはず
        cands = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if not movable_arr[dx, dy]:
                    continue
                if is_black:
                    new_Bx = Bx + dx
                    new_By = By + dy
                    new_Wx = Wx
                    new_Wy = Wy
                    if new_Bx == new_Wx and new_By == new_Wy:  # 相手のいる方向に進む場合、乗り越える
                        new_Bx = new_Bx + dx
                        new_By = new_By + dy
                    new_d = dist_array1[new_Bx, new_By]
                else:
                    new_Bx = Bx
                    new_By = By
                    new_Wx = Wx + dx
                    new_Wy = Wy + dy
                    if new_Bx == new_Wx and new_By == new_Wy:  # 相手のいる方向に進む場合、乗り越える
                        new_Wx = new_Wx + dx
                        new_Wy = new_Wy + dy
                    new_d = dist_array2[new_Wx, new_Wy]
                
                if new_d < d and (new_Bx, new_By, new_Wx, new_Wy, not is_black) in optimal_dict.keys():
                    cands.append((new_Bx, new_By, new_Wx, new_Wy, not is_black, optimal_dict[(new_Bx, new_By, new_Wx, new_Wy, not is_black)][5], True))
                else:
                    cands.append((new_Bx, new_By, new_Wx, new_Wy, not is_black, 0, False))  # 距離を縮められないのは有効でないとする

        # candsがないのは壁に取り囲まれてまったく身動きが取れない場合のみ。それは解を求めなくても良い
        if len(cands) == 0:
            continue

        if is_black:
            cands = sorted(cands, key=lambda x: x[5] - (not x[6]) * 100, reverse=True)  # Bd-Wd最大化
        else:
            cands = sorted(cands, key=lambda x: -x[5] - (not x[6]) * 100, reverse=True)  # Bd-Wd最小化

        max_score = cands[0][5]
        for i, cand in enumerate(cands):
            if cand[5] < max_score:
                max_score_index = i
                break
            else:
                max_score_index = i + 1

        optimal_dict[(Bx, By, Wx, Wy, is_black)] = cands[random.randrange(max_score_index)]  # 一番良い行動を記録

    return optimal_dict


def calc_next_state(x):
    state, action = x
    accept_action_str(state, actionid2str(state, action), check_placable=False)
    return state


class BasicAI(Agent):
    def __init__(self, color, search_nodes=1, C_puct=5, tau=1, n_parallel=N_PARALLEL, virtual_loss_n=1, use_estimated_V=True, V_ema_w=0.01, 
                 shortest_only=False, use_average_Q=False, random_playouts=False, tau_mult=2, tau_decay=6, is_mimic_AI=False, tau_peak=6, force_opening=None):
        super(BasicAI, self).__init__(color)
        self.search_nodes = search_nodes
        self.C_puct = C_puct
        self.tau = tau
        self.n_parallel = n_parallel
        self.virtual_loss_n = virtual_loss_n
        self.use_estimated_V = use_estimated_V
        self.use_average_Q = use_average_Q
        self.V_ema_w = V_ema_w
        self.init_prev()
        self.tree_for_visualize = None
        self.shortest_only = shortest_only
        self.random_playouts = random_playouts
        self.tau_mult = tau_mult
        self.tau_decay = tau_decay
        self.is_mimic_AI = is_mimic_AI  # 指定された後手AIは、先手を追い越すまでの間、回転対称になるように先手の手を真似する
        self.tau_peak = tau_peak  # ランダム性を一番高くするターン数。初手に壁置くのを読むのは無駄だが、向かい合ったときにいろいろな壁の起き方をするのは有意義だろうという考え
        self.force_opening = force_opening

    def init_prev(self, state=None):
        # 試合前に毎回実行
        if state is None:
            state = State()
            State_init(state)
        # p = np.ones((137,)) / 137.
        # p = np.asarray(p, dtype=np.float32)
        self.prev_tree = None
        self.prev_action = None
        self.estimated_V = 0.0
        self.prev_wall_num = state.black_walls + state.white_walls
        self.init_discovery(state)
        self.optimal_dict = None
        self.end_game = False  # 移動のみで十分な局面
        self.end_mimic = False  # 231208時点ではmimic_AIのときのみTrueになりうる

        # if self.state2node_per_turn is not None:
        #     for v in self.state2node_per_turn.values():
        #         del v
        self.state2node_per_turn = {}  # turn -> (state.vec -> node)

    def get_state_vec_and_is_state_searched(self, tree):
        turn = tree.s.turn
        state_vec = get_state_vec_from_tree(tree)

        if turn not in self.state2node_per_turn.keys():
            return state_vec, False

        if state_vec in self.state2node_per_turn[turn].keys():
            return state_vec, True
        
        return state_vec, False
        
    def add_state2node_per_turn_item(self, turn, state_vec, node):
        if turn not in self.state2node_per_turn.keys():
            self.state2node_per_turn[turn] = {}

        self.state2node_per_turn[turn][state_vec] = node
        
    def init_discovery(self, state):

        # 探索済みマスをTrueにする配列、0にB, 1にWを割当て
        self.discovery_arr = np.zeros((2, 9, 9), dtype=bool)
        for color in range(2):
            x, y = color_p(state, color)
            self.discovery_arr[color, x, y] = True

    def act(self, state, showNQ=False, noise=0., use_prev_tree=True, opponent_prev_tree=None, return_root_tree=False):
        #tau = (1 + (self.tau_mult - 1) * np.power(0.5, state.turn / self.tau_decay)) * self.tau
        tau = (1 + (self.tau_mult - 1) * np.exp(- np.square((state.turn - self.tau_peak) / self.tau_decay))) * self.tau
        if return_root_tree:
            action_id, tree = self.MCTS(state, self.search_nodes, self.C_puct, tau, showNQ, noise, use_prev_tree=use_prev_tree, opponent_prev_tree=opponent_prev_tree, return_root_tree=True)
            return action_id, tree
        else:
            action_id, _, _, _, _ = self.MCTS(state, self.search_nodes, self.C_puct, tau, showNQ, noise, use_prev_tree=use_prev_tree, opponent_prev_tree=opponent_prev_tree)
            return action_id

    def act_and_get_pi(self, state, showNQ=False, noise=0., use_prev_tree=True, opponent_prev_tree=None):
        #tau = (1 + (self.tau_mult - 1) * np.power(0.5, state.turn / self.tau_decay)) * self.tau
        tau = (1 + (self.tau_mult - 1) * np.exp(- np.square((state.turn - self.tau_peak) / self.tau_decay))) * self.tau
        action_id, pi, v_prev, v_post, searched_node_num = self.MCTS(state, self.search_nodes, self.C_puct, tau, showNQ, noise, use_prev_tree=use_prev_tree, opponent_prev_tree=opponent_prev_tree)
        return action_id, pi, v_prev, v_post, searched_node_num

    def action_array(self, s):
        if s.terminate:
            return np.zeros((2 * (State.BOARD_LEN - 1) * (State.BOARD_LEN - 1) + 9,), dtype="bool")
        r, c = placable_flatten_array(s, s.turn % 2)
        x, y = color_p(s, s.turn % 2)
        v = np.concatenate([r, c, movable_array_flatten(s, x, y)])
        return np.asarray(v, dtype="bool")

    def movable_array(self, x, y, s):
        if self.shortest_only:
            return movable_array_flatten(s, x, y, shortest_only=True)

        # 壁が両方0なら最短路以外の手はない。教師の質向上を目的とする
        if s.black_walls == 0 and s.white_walls == 0:
            return movable_array_flatten(s, x, y, shortest_only=True)

        ret = movable_array(s, x, y, shortest_only=True)  # 距離を縮める方向には必ず動けるとする

        all_movable_arr = movable_array(s, x, y, shortest_only=False)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x2 = x + dx
                y2 = y + dy
                if x2 < 0 or x2 >= 9 or y2 < 0 or y2 >= 9:
                    continue

                # 移動可能かつ未探索なら追加
                if all_movable_arr[dx, dy] and not self.discovery_arr[s.turn % 2, x2, y2]:
                    ret[dx, dy] = True

        ret = ret.flatten()

        return ret

    def select(self, root_tree, C_puct):
        t = root_tree
        a = 0
        nodes = []
        actions = []
        while True:
            
            if t.P is None:
                print("!"*200)
                print(actions)
                assert False, "t.P is None is not expected"

            # 負けノードは探索しない
            a = select_action(t.tree_c.contents.Q_arr, t.tree_c.contents.N_arr, t.P_without_loss.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              C_puct, self.estimated_V, self.color, t.s.turn)

            nodes.append(t)
            actions.append(a)

            if a not in t.children.keys():
                return t, a, nodes, actions, False # 葉ノードでない
            else:
                t = t.children[a]

    def calc_leaf_movable_arr(self, root_state, actions):
        # stateはroot、actionsはMCTSのleafまでの道のりを想定。actionsはどれも合法手とする。
        leaf_discovery_arr = np.copy(self.discovery_arr)
        state = state_copy(root_state)
        for i, action in enumerate(actions):
            try:
                # checkしないため、内部的には非合法手扱いされることがありFalseになることがある。ただし、actionsが合法手からなるので問題なし。
                accept_action_str(state, actionid2str(state, action), check_placable=False, calc_placable_array=False, check_movable=False)
            except:
                print("{} error action={}".format(i, action))
                print(actions)
                display_cui(root_state)
                display_cui(state)
                print("!"*30)
            if action < 128:
                leaf_discovery_arr = np.zeros((2, 9, 9), dtype=bool)
            leaf_discovery_arr[0, state.Bx, state.By] = True
            leaf_discovery_arr[1, state.Wx, state.Wy] = True

        x, y = color_p(state, state.turn % 2)
        ret = movable_array(state, x, y, shortest_only=True)  # 距離を縮める方向には必ず動けるとする
        if self.shortest_only or (state.black_walls == 0 and state.white_walls == 0):
            return ret.flatten()

        all_movable_arr = movable_array(state, x, y, shortest_only=False)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x2 = x + dx
                y2 = y + dy
                if x2 < 0 or x2 >= 9 or y2 < 0 or y2 >= 9:
                    continue

                # 移動可能かつ未探索なら追加
                if all_movable_arr[dx, dy] and not leaf_discovery_arr[state.turn % 2, x2, y2]:
                    ret[dx, dy] = True
        ret = ret.flatten()

        return ret

    def MCTS(self, state, max_node, C_puct, tau, showNQ=False, noise=0., random_flip=False, use_prev_tree=True, opponent_prev_tree=None, return_root_tree=False):
        if self.random_playouts:
            max_node = SELFPLAY_SEARCHNODES_MIN
            if random.random() < DEEP_SEARCH_P:
                max_node = SELFPLAY_SEARCHNODES_MAX
            node_num_expectation = DEEP_SEARCH_P * SELFPLAY_SEARCHNODES_MAX + (1 - DEEP_SEARCH_P) * SELFPLAY_SEARCHNODES_MIN
        else:
            node_num_expectation = max_node
        
        root_v = self.v(state)[0]

        wall_num = state.black_walls + state.white_walls

        B_dist, W_dist = get_player_dist_from_goal(state)

        # 自分の道が確定していて相手よりも早く着くなら最短路を進むだけ
        if is_certain_path_terminate(state):
            self.tree_for_visualize = self.prev_tree = None  # 読みが入らなくなるので、prev_treeが参照されないようにNoneを入れておく
            x, y = color_p(state, state.turn % 2)
            movable_arr = movable_array(state, x, y, shortest_only=True)
            if not np.any(movable_arr):
                movable_arr = movable_array(state, x, y, shortest_only=False)
            cands = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if movable_arr[dx, dy]:
                        cands.append((dx, dy))
            dx, dy = random.choice(cands)
            if dx == -1:
                dx = 2
            if dy == -1:
                dy = 2
            action_id = 128 + dx * 3 + dy
            pi_ret = np.zeros((137))
            pi_ret[action_id] = 1.0
            self.prev_wall_num = wall_num
            if return_root_tree:
                return action_id, Tree(state, pi_ret)
            else:
                return action_id, pi_ret, root_v, 0.0, node_num_expectation  # 探索はしていないので探索後のvは0にしておく

        if wall_num == 0 or (state.black_walls == 0 and (W_dist + (1 - state.turn % 2) <= B_dist - 1)) or (state.white_walls == 0 and (B_dist + state.turn % 2 <= W_dist - 1)):
            self.end_game = True

        if self.end_game:
            #print("endgame")
            self.tree_for_visualize = self.prev_tree = None  # 読みが入らなくなるので、prev_treeが参照されないようにNoneを入れておく
            if (self.optimal_dict is None) or (self.prev_wall_num != wall_num):
                #print("DP")
                self.optimal_dict = calc_optimal_move_by_DP(state)
            Bx, By, Wx, Wy = state.Bx, state.By, state.Wx, state.Wy
            is_black = (state.turn % 2 == 0)
            next_Bx, next_By, next_Wx, next_Wy, _, _, _ = self.optimal_dict[(Bx, By, Wx, Wy, is_black)]
            if self.color == 0:
                dx = next_Bx - Bx
                dy = next_By - By
            else:
                dx = next_Wx - Wx
                dy = next_Wy - Wy
            action_id = dxdy2actionid(dx, dy)
            #print(dx, dy)
            #print(actionid2str(state, action_id))
            pi_ret = np.zeros((137))
            pi_ret[action_id] = 1.0
            self.prev_wall_num = wall_num
            if return_root_tree:
                return action_id, Tree(state, pi_ret)
            else:
                return action_id, pi_ret, root_v, 0.0, node_num_expectation  # 探索はしていないので探索後のvは0にしておく

        if wall_num != self.prev_wall_num:
            self.init_discovery(state)
        for color in range(2):
            x, y = color_p(state, color)
            self.discovery_arr[color, x, y] = True

        x, y = color_p(state, state.turn % 2)
        p = self.p(state, leaf_movable_arrs=[self.movable_array(x, y, state)])
        illegal = (p == 0.)
        p = (1. - noise) * p + noise * np.random.dirichlet([0.3] * len(p))  # ディリクレ分布はチェスと同じ設定
        p[illegal] = 0.
        p = p / sum(p)
        p = np.asarray(p, dtype=np.float32)

        visited = set([])

        def negate_tree(tree, count=0):
            state_vec = get_state_vec_from_tree(tree)
            if state_vec in visited:
                return tree
            
            visited.add(state_vec)

            ret = Tree(tree.s, tree.P)

            ret.V = -tree.V
            ret.result = tree.result  # resultは立場によらずに定まるからそのままコピー
            ret.optimal_action = tree.optimal_action  
            ret.is_lose_child_arr = tree.is_lose_child_arr
            ret.P_without_loss = tree.P_without_loss
            ret.dist_diff_arr = tree.dist_diff_arr
            ret.already_certain_path_confirmed = tree.already_certain_path_confirmed
            ret.state_vec = tree.state_vec
            ret.arrays_for_feature_CNN = tree.arrays_for_feature_CNN

            copy_int_arr(ret.tree_c.contents.N_arr, tree.tree_c.contents.N_arr)
            copy_float_arr(ret.tree_c.contents.W_arr, tree.tree_c.contents.W_arr, -1)
            copy_float_arr(ret.tree_c.contents.Q_arr, tree.tree_c.contents.Q_arr, -1)
            assert count <= max_node, "negate_treeで無限再帰の可能性" 
            for key, child_tree in tree.children.items():
                ret.children[key] = negate_tree(child_tree, count + 1)
            return ret

        if use_prev_tree:
            if self.prev_tree is not None and opponent_prev_tree is None:
                root_tree = self.prev_tree
                root_tree.s = state
                if self.prev_action is not None and self.prev_action in root_tree.children.keys() and state.turn == root_tree.s.turn:
                    root_tree = root_tree.children[self.prev_action]
                else:
                    root_tree = Tree(state, p)
                    self.state2node_per_turn = {}
            elif opponent_prev_tree is not None:
                if self.prev_tree is not None:
                    del self.prev_tree
                root_tree = negate_tree(opponent_prev_tree)
            else:
                root_tree = Tree(state, p)
                self.state2node_per_turn = {}
        else:
            root_tree = Tree(state, p)
            self.state2node_per_turn = {}

        # 載せたノイズを反映させる
        root_tree.set_P(p)

        # 非合法手のNを強制的に0にして、例えば探索済みマスに戻るような手を読まないようにする
        mult_int_arr(root_tree.tree_c.contents.N_arr, np.array(~illegal, dtype="int32").ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        mult_float_arr(root_tree.tree_c.contents.W_arr, np.array(~illegal, dtype="float32").ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        mult_float_arr(root_tree.tree_c.contents.Q_arr, np.array(~illegal, dtype="float32").ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        # def add_virtual_loss(node, action):
        #     node.tree_c.contents.N_arr[action] += self.virtual_loss_n
        #     if self.color == node.s.turn % 2:  # 先後でQがひっくり返ることを考慮
        #         node.tree_c.contents.W_arr[action] -= self.virtual_loss_n
        #     else:
        #         node.tree_c.contents.W_arr[action] += self.virtual_loss_n
        #     node.tree_c.contents.Q_arr[action] = node.tree_c.contents.W_arr[action] / node.tree_c.contents.N_arr[action]

        def should_deepsearch(W, N, root_v):
            return self.random_playouts and abs(sum(W) / sum(N) - root_v) >= DEEP_TH

        node_num = sum(root_tree.tree_c.contents.N_arr)

        if node_num >= max_node - self.n_parallel and should_deepsearch(root_tree.tree_c.contents.W_arr, root_tree.tree_c.contents.N_arr, root_v):
            max_node = SELFPLAY_SEARCHNODES_MAX

        # force_openingがある場合、合法手である限りは必ず指す
        if self.force_opening is not None and state.turn < len(self.force_opening):
            force_action = Official2Glendenning(self.force_opening[state.turn])
            force_action_id = str2actionid(state, force_action)
            if not illegal[force_action_id]:
                pi_ret = np.zeros((137))
                pi_ret[force_action_id] = 1.0
                self.prev_wall_num = wall_num
                self.prev_tree = None
                if return_root_tree:
                    return force_action_id, Tree(state, pi_ret)
                else:
                    return force_action_id, pi_ret, root_v, 0.0, 1  # 探索はしていないので探索後のvは0にしておく、またpolicyの学習への影響を最小限にする

        while node_num < max_node and root_tree.result == 0:
            # select
            nodess = []
            actionss = []

            for _ in range(min(self.n_parallel, max_node)):
                _, _, nodes, actions, _ = self.select(root_tree, C_puct)
                if nodes is None:
                    break
                nodess.append(nodes)
                actionss.append(actions)

                for node, action in zip(nodes, actions):
                    if self.color == node.s.turn % 2:
                        coef = -1
                    else:
                        coef = 1
                    add_virtual_loss(node.tree_c, action, self.virtual_loss_n, coef)

            # virtual lossを元に戻す
            for nodes, actions in zip(nodess, actionss):
                for node, action in zip(nodes, actions):
                    node.tree_c.contents.N_arr[action] -= self.virtual_loss_n
                    if self.color == node.s.turn % 2:
                        node.tree_c.contents.W_arr[action] += self.virtual_loss_n
                    else:
                        node.tree_c.contents.W_arr[action] -= self.virtual_loss_n
                    if node.tree_c.contents.N_arr[action] == 0:
                        node.tree_c.contents.Q_arr[action] = 0.
                    else:
                        node.tree_c.contents.Q_arr[action] = node.tree_c.contents.W_arr[action] / node.tree_c.contents.N_arr[action]

            states = []
            leaf_movable_arrs = []
            arrss_for_feature_CNN = []
            for nodes, actions in zip(nodess, actionss):
                leaf_movable_arrs.append(self.calc_leaf_movable_arr(state, actions))
                s = state_copy(nodes[-1].s)
                #print([self.actionid2str(node.s, action) for node, action in zip(nodes, actions)])
                actionstr = actionid2str(s, actions[-1])
                accept_action_str(s, actionstr, check_placable=False)  # 合法手チェックしないことで高速化。actionsに非合法手が含まれないことが前提。
                states.append(s)
                
                if len(actionstr) == 2:  # 移動なら、行動前の状態の配列を再利用する（NoneはNoneのまま格納）
                    arrss_for_feature_CNN.append(nodes[-1].arrays_for_feature_CNN)
                else:
                    arrss_for_feature_CNN.append(None)
            node_num += len(states)

            p_arr, v_arr = self.pv_array(states, arrss_for_feature_CNN, leaf_movable_arrs)

            # for nodes2, actions2 in zip(nodess, actionss):
            #     pass
            #     print([Glendenning2Official(actionid2str(node.s, action)) for node, action in zip(nodes2, actions2)])
            # print("")

            # expand
            for count, s, nodes, actions, arrs_for_feature_CNN in zip(range(len(states)), states, nodess, actionss, arrss_for_feature_CNN):
                if not s.pseudo_terminate:
                    # sはtでaを実行したときのstate
                    t = nodes[-1]
                    a = actions[-1]
                    if a in t.children.keys():  # 過去に探索済み
                        continue

                    new_tree = Tree(s, None)
                    state_vec, is_searched = self.get_state_vec_and_is_state_searched(new_tree)
                    if is_searched:  # 過去に探索していた状態と一致したとき  
                        t.children[a] = self.state2node_per_turn[s.turn][state_vec]
                        # t.children[a].set_P(np.array(p_arr[count], dtype=np.float32))
                        # t.children[a].V = np.array(v_arr[count], dtype=np.float32)
                        if s.turn != t.children[a].s.turn or t.s.turn + 1 != t.children[a].s.turn:
                            display_cui(t.s)
                            display_cui(s)
                            t.children[a].s
                            assert False, "!" * 100

                    else:  # 初めて探索されたとき
                        t.children[a] = new_tree
                        t.children[a].set_P(np.array(p_arr[count], dtype=np.float32))
                        t.children[a].V = np.array(v_arr[count], dtype=np.float32)
                        self.add_state2node_per_turn_item(s.turn, state_vec, t.children[a])
                        t.children[a].arrays_for_feature_CNN = arrs_for_feature_CNN

            # backup
            count = 0
            for nodes, actions, s in zip(nodess, actionss, states):
                for node, action in zip(nodes, actions):
                    node.tree_c.contents.N_arr[action] += 1
                    node.tree_c.contents.W_arr[action] += v_arr[count]
                    self.estimated_V = self.estimated_V * (1 - self.V_ema_w) + v_arr[count] * self.V_ema_w
                    node.tree_c.contents.Q_arr[action] = node.tree_c.contents.W_arr[action] / node.tree_c.contents.N_arr[action]
                count += 1

            if node_num >= max_node - self.n_parallel and should_deepsearch(root_tree.tree_c.contents.W_arr, root_tree.tree_c.contents.N_arr, root_v):
                max_node = SELFPLAY_SEARCHNODES_MAX

            # 勝敗ノード決定
            for count, s, nodes, actions in zip(range(len(states)), states, nodess, actionss):
                # 葉ノードに当たったときのみ変更がありえるので葉ノード以外除外
                if not s.pseudo_terminate:
                    continue

                # 葉ノード側からたどる
                for node, action in zip(nodes[::-1], actions[::-1]):
                    # 過去の探索で既に勝敗が決まっているときは飛ばす(optimal_actionなどが書き換えられないよう)
                    if node.result != 0:
                        continue

                    is_win_node = (node.s.turn % 2 == 0 and s.pseudo_reward == 1) or (node.s.turn % 2 == 1 and s.pseudo_reward == -1)
                    win_reward = 1 if node.s.turn % 2 == 0 else -1
                    lose_reward = -1 if node.s.turn % 2 == 0 else 1

                    if action not in node.children.keys():  # 葉ノードを子に持つ
                        s_B_dist, s_W_dist = get_player_dist_from_goal(s)
                        node.dist_diff_arr[action] = max(s_W_dist - s_B_dist + (1 - s.turn % 2), s_B_dist - s_W_dist + s.turn % 2)
                    elif  node.children[action].result != 0:  # 勝敗が決定した子ノードを持つ
                        node.dist_diff_arr[action] = int(np.min(node.children[action].dist_diff_arr))

                    if is_win_node:  # 勝ち側は一つでも勝ちに向かう行動があれば勝ちノード
                        if action not in node.children.keys() or node.children[action].result == win_reward:
                            node.result = s.pseudo_reward
                            node.optimal_action = action
                            node_illegal = (node.P == 0.)
                            
                            if node_illegal[action]:
                                print("!"*100)
                                display_cui(state)
                                display_cui(node.s)
                                print(node.P)
                                print(node_illegal)
                                print(actionid2str(node.s, action))
                                print(actions)
                            assert not node_illegal[action], actionid2str(node.s, action)
                            
                    else:  # 負け側はすべての行動が相手の勝ちになるときに限り負けノード
                        if action not in node.children.keys() or node.children[action].result == lose_reward:
                            node.set_is_lose_child_arr(action, True)

                            if not node.already_certain_path_confirmed: # 壁置きで負けになる場合、一度確定路判定をする。移動の場合も判定するとだいぶ遅くなるので壁おきに制限。
                                node.already_certain_path_confirmed = True
                                if is_certain_path_terminate(node.s, (node.s.turn + 1) % 2):  # 負け側ノードが壁置きをしなくても既に相手が確定路により勝ちの場合は、任意の壁置きで負けになることがわかる。
                                    node.set_is_lose_child_arr_True(np.arange(128))
                            node_illegal = (node.P == 0.)
                            if np.all(node_illegal | node.is_lose_child_arr):
                                node.result = s.pseudo_reward
                                node.optimal_action = int(np.argmin(node.dist_diff_arr))
                            else:
                                break
                
            # root_treeの勝敗が決まったら探索を打ち切る
            if root_tree.result != 0:
                break

        if showNQ:
            print("p=")
            display_parameter(np.asarray(root_tree.P * 1000, dtype="int32"))
            print("N=")
            display_parameter(np.asarray(root_tree.tree_c.contents.N_arr, dtype="int32"))
            print("Q=")
            display_parameter(np.asarray(np.array(root_tree.tree_c.contents.Q_arr) * 1000, dtype="int32"))
            print("prev v={:.3f}, post v={:.3f}".format(root_v, sum(root_tree.tree_c.contents.W_arr) / sum(root_tree.tree_c.contents.N_arr)))
            print("root_tree result = {}".format(root_tree.result))

        if root_tree.result != 0:  # 勝敗決定の場合
            action = root_tree.optimal_action
            pi = np.zeros((137))
            pi[action] = 1.0
        else:
            x, y = color_p(state, state.turn % 2)
            shortest_move = movable_array_flatten(state, x, y, shortest_only=True)
            move_N = np.array(root_tree.tree_c.contents.N_arr[128:])
            move_Q = np.array(root_tree.tree_c.contents.Q_arr[128:])
            use_shortest = (move_N >= int(sum(root_tree.tree_c.contents.N_arr) * SHORTEST_N_RATIO)) & (move_Q >= SHORTEST_Q)  # 十分探索していて、十分勝ちに近い手なら、できる限り最短路を選ぶことで試合を早く終わらせる
            use_shortest = use_shortest & shortest_move

            N2 = np.array(root_tree.tree_c.contents.N_arr)
            if np.any(use_shortest):
                N2[128:] = move_N * use_shortest

            if tau == 0:
                N2 = N2 * (N2 == np.max(N2))
            else:
                N2 = np.power(np.asarray(N2, dtype="float64"), 1. / tau)
            pi = N2 / np.sum(N2)
            action = np.random.choice(len(pi), p=pi)

        if self.is_mimic_AI:
            # print("mimic AI")

            # actionの座標を計算し、回転対称に写す。
            if self.prev_action is not None:
                is_placewall = (self.prev_action < 128)
                if is_placewall:
                    is_row = self.prev_action < 64
                    wall_id = self.prev_action % 64
                    prev_wall_x = wall_id // 8
                    prev_wall_y = wall_id % 8
                    wall_x = 7 - prev_wall_x
                    wall_y = 7 - prev_wall_y
                    action_id = int(not is_row) * 64 + wall_x * 8 + wall_y
                    # print(is_row, wall_x, wall_y, action_id) 
                else:
                    move_id = self.prev_action - 128
                    prev_move_x, prev_move_y = move_id2dxdy(move_id)
                    move_x = -prev_move_x
                    move_y = -prev_move_y
                    action_id = dxdy2actionid(move_x, move_y)
                    # print(move_x, move_y, action_id)

                end_mimic = state.turn >= FORCE_MIMIC_TURN and root_tree.tree_c.contents.N_arr[action_id] / sum(root_tree.tree_c.contents.N_arr) <= MIMIC_N_RATIO
                
                # 合法手でない手を打たないように対策
                if root_tree.tree_c.contents.N_arr[action_id] == 0:
                    end_mimic = True

                if not self.end_mimic and not end_mimic:
                    action = action_id

                if is_jump_move(state, action_id) or end_mimic:
                    self.end_mimic = True

        # 葉に向う行動は勝ちになる行動のみ
        if action in root_tree.children.keys():
            self.prev_tree = root_tree.children[action]
        else:
            self.prev_tree = None
        self.tree_for_visualize = root_tree
        self.prev_wall_num = wall_num

        if return_root_tree:
            return action, root_tree
        elif root_tree.result != 0:  # 勝敗決定の場合
            
            return action, pi, root_v, (sum(root_tree.tree_c.contents.W_arr) + root_v) / (sum(root_tree.tree_c.contents.N_arr) + 1), node_num_expectation
        else:
            return action, np.array(root_tree.tree_c.contents.N_arr) / sum(root_tree.tree_c.contents.N_arr), root_v, (sum(root_tree.tree_c.contents.W_arr) + root_v) / (sum(root_tree.tree_c.contents.N_arr) + 1), sum(root_tree.tree_c.contents.N_arr)

    def get_tree_for_graphviz(self):
        if self.tree_for_visualize is None:
            return None
        
        g = Digraph(format='png')
        g.attr('node', shape='circle')

        config = read_application_config()

        get_graphviz_tree_for_shared_tree(self.tree_for_visualize, g, threshold=config["graphviz_threshold"])

        return g
