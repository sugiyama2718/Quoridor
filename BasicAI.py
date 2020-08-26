# coding:utf-8
from Agent import Agent, actionid2str
from Tree import Tree
import State
import numpy as np
import copy
from graphviz import Digraph
import os
#from multiprocessing import Pool
#import multiprocessing as multi
#from joblib import Parallel, delayed

num2str = {0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f", 6:"g", 7:"h", 8:"i"}


# 引数のgにgraphviz用のグラフを入れる
def get_graphviz_tree(tree, g, count=0, threshold=5):
    if len(tree.children.items()) == 0:
        g.node(str(count), label="0")
    else:
        parent_count = count
        g.node(str(parent_count), label=str(int(np.sum(tree.N))) + os.linesep + "{:.3f}".format(np.sum(tree.W) / np.sum(tree.N)))
        count += 1
        for key, value in tree.children.items():
            if int(tree.N[key]) >= threshold:
                g.edge(str(parent_count), str(count), label=actionid2str(tree.s, key) + os.linesep + str(int(tree.N[key])))
                get_graphviz_tree(value, g, count)
            count += int(np.sum(value.N)) + 1


def state_copy(s):
    ret = State.State()
    ret.seen = copy.copy(s.seen)
    ret.row_wall = copy.copy(s.row_wall)
    ret.column_wall = copy.copy(s.column_wall)
    ret.must_be_checked_x = copy.copy(s.must_be_checked_x)
    ret.must_be_checked_y = copy.copy(s.must_be_checked_y)
    ret.placable_rb = copy.copy(s.placable_rb)
    ret.placable_cb = copy.copy(s.placable_cb)
    ret.placable_rw = copy.copy(s.placable_rw)
    ret.placable_cw = copy.copy(s.placable_cw)
    ret.Bx = s.Bx
    ret.By = s.By
    ret.Wx = s.Wx
    ret.Wy = s.Wy
    ret.turn = s.turn
    ret.black_walls = s.black_walls
    ret.white_walls = s.white_walls
    ret.terminate = s.terminate
    ret.reward = s.reward
    ret.row_special_cut = s.row_special_cut
    ret.row_eq = s.row_eq
    ret.row_graph = s.row_graph
    ret.column_special_cut = s.column_special_cut
    ret.column_eq = s.column_eq
    ret.column_graph = s.column_graph
    ret.dist_array1 = s.dist_array1
    ret.dist_array2 = s.dist_array2
    return ret


class BasicAI(Agent):
    def __init__(self, color, search_nodes=1, C_puct=5, tau=1, n_parallel=8, virtual_loss_n=1):
        super(BasicAI, self).__init__(color)
        self.search_nodes = search_nodes
        self.C_puct = C_puct
        self.tau = tau
        self.n_parallel = n_parallel
        self.virtual_loss_n = virtual_loss_n
        self.init_prev()
        self.tree_for_visualize = None

    def init_prev(self):
        state = State.State()
        p = np.ones((137,)) / 137.
        self.prev_tree = Tree(state, p)
        self.prev_action = None

    def act(self, state, showNQ=False, noise=0.):
        action_id, _ = self.MCTS(state, self.search_nodes, self.C_puct, self.tau, showNQ, noise)
        return action_id

    def act_and_get_pi(self, state, showNQ=False, noise=0.):
        action_id, pi = self.MCTS(state, self.search_nodes, self.C_puct, self.tau, showNQ, noise)
        return action_id, pi

    def action_array(self, s):
        if s.terminate:
            return np.zeros((2 * (State.BOARD_LEN - 1) * (State.BOARD_LEN - 1) + 9,), dtype="bool")
        r, c = s.placable_array(s.turn % 2)
        x, y = s.color_p(s.turn % 2)
        v = np.concatenate([r.flatten(), c.flatten(), s.movable_array(x, y).flatten()])
        return np.asarray(v, dtype="bool")

    def display_parameter(self, x):
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

    def select(self, root_tree, C_puct):
        t = root_tree
        a = 0
        nodes = []
        actions = []
        count = 0
        while True:
            if t.P is None:
                t.P = self.p_array([t.s])[0]

            if self.color == t.s.turn % 2:
                x = t.Q + C_puct * t.P * np.sqrt(1. + np.sum(t.N)) / (1. + t.N)
            else:
                x = -t.Q + C_puct * t.P * np.sqrt(1. + np.sum(t.N)) / (1. + t.N)
            x -= np.min(x)
            #x[~self.action_array(t.s)] = -1.
            x[t.P == 0.] = -1.  # できない行動を選択しない（できない行動はp=0となることが前提）

            a = np.argmax(x)

            nodes.append(t)
            actions.append(a)

            if a not in t.children.keys():
                break
            else:
                t = t.children[a]
            count += 1
        return t, a, nodes, actions

    def MCTS(self, state, max_node, C_puct, tau, showNQ=False, noise=0., random_flip=False):
        # 壁がお互いになく、分岐のない場合読みを入れない。ただし、それでもprev_treeとかの関係上振る舞いが変わるので保留中。
        #search_node_num = max_node
        #if state.black_walls == 0 and state.white_walls == 0:
        #    x, y = state.color_p(state.turn % 2)
        #    if int(np.sum(state.movable_array(x, y, shortest_only=True))) == 1:
        #        search_node_num = 1
        p = self.p(state)
        illegal = (p == 0.)
        old_p = p
        p = (1. - noise) * p + noise * np.random.rand(len(p))
        p[illegal] = 0.
        p = p / sum(p)
        #root_tree = Tree(state, p)
        root_tree = self.prev_tree
        root_tree.s = state
        if self.prev_action is not None:
            if self.prev_action in root_tree.children.keys():
                root_tree = root_tree.children[self.prev_action]
            else:
                root_tree = Tree(state, p)
        root_tree.P = p

        node_num = np.sum(root_tree.N)
        while node_num < max_node:
            # select
            nodess = []
            actionss = []
            for j in range(min(self.n_parallel, max_node)):
                _, _, nodes, actions = self.select(root_tree, C_puct)
                if nodes is None:
                    break
                nodess.append(nodes)
                actionss.append(actions)

                # virtual loss
                for node, action in zip(nodes, actions):
                    node.N[action] += self.virtual_loss_n
                    if self.color == node.s.turn % 2:  # 先後でQがひっくり返ることを考慮
                        node.W[action] -= self.virtual_loss_n
                    else:
                        node.W[action] += self.virtual_loss_n
                    node.Q[action] = node.W[action] / node.N[action]
            for nodes, actions in zip(nodess, actionss):
                # virtual lossを元に戻す
                for node, action in zip(nodes, actions):
                    node.N[action] -= self.virtual_loss_n
                    if self.color == node.s.turn % 2:
                        node.W[action] += self.virtual_loss_n
                    else:
                        node.W[action] -= self.virtual_loss_n
                    if node.N[action] == 0:
                        node.Q[action] = 0.
                    else:
                        node.Q[action] = node.W[action] / node.N[action]

            states = []
            for nodes, actions in zip(nodess, actionss):
                s = state_copy(nodes[-1].s)
                #print([self.actionid2str(node.s, action) for node, action in zip(nodes, actions)])
                s.accept_action_str(actionid2str(s, actions[-1]))
                states.append(s)
            node_num += len(states)

            #p = self.p_array(states, random_flip=random_flip)
            v = self.v_array(states, random_flip=random_flip)

            for nodes2, actions2 in zip(nodess, actionss):
                pass
                #print([self.actionid2str(node.s, action) for node, action in zip(nodes2, actions2)])
            #print("")

            count = 0
            for s, nodes, actions in zip(states, nodess, actionss):
                if not s.terminate:
                    t = nodes[-1]
                    a = actions[-1]
                    if a not in t.children.keys():
                        t.children[a] = Tree(s, None)
                count += 1

            # backup
            count = 0
            for nodes, actions in zip(nodess, actionss):
                for node, action in zip(nodes, actions):
                    node.N[action] += 1
                    node.W[action] += v[count]
                    node.Q[action] = node.W[action] / node.N[action]
                count += 1
        if showNQ:
            print("p=")
            self.display_parameter(np.asarray(old_p * 1000, dtype="int32"))
            print("N=")
            self.display_parameter(np.asarray(root_tree.N, dtype="int32"))
            print("Q=")
            self.display_parameter(np.asarray(root_tree.Q * 1000, dtype="int32"))
            print("v={}".format(self.v(root_tree.s)))

        if tau == 0:
            N2 = root_tree.N * (root_tree.N == np.max(root_tree.N))
        else:
            N2 = np.power(np.asarray(root_tree.N, dtype="float64"), 1. / tau)
        pi = N2 / np.sum(N2)
        action = np.random.choice(len(pi), p=pi)
        # 葉に向う行動は勝ちになる行動のみ
        if action in root_tree.children.keys():
            self.prev_tree = root_tree.children[action]
        action2 = np.argmax(root_tree.N)
        pi_ret = np.zeros((137, ))
        pi_ret[action2] = 1.
        self.tree_for_visualize = root_tree
        return action, root_tree.N / np.sum(root_tree.N)

    def get_tree_for_graphviz(self):
        g = Digraph(format='png')
        g.attr('node', shape='circle')

        get_graphviz_tree(self.tree_for_visualize, g)

        return g





