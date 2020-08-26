# coding:utf-8
import numpy as np


class Tree:
    # p is prior probability
    # pにはNoneが来ても良い。その場合必要なときに代入するべきことを表す。
    def __init__(self, s, p):
        #action_n = p.shape[0]
        action_n = 137
        self.children = {}
        self.s = s
        self.N = np.zeros((action_n,))
        self.W = np.zeros((action_n,))
        self.Q = np.zeros((action_n,))
        self.P = p