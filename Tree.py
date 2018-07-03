import numpy as np


class Tree:
    # p is prior probability
    def __init__(self, s, p):
        action_n = p.shape[0]
        self.children = {}
        self.s = s
        self.N = np.zeros((action_n,))
        self.W = np.zeros((action_n,))
        self.Q = np.zeros((action_n,))
        self.P = p