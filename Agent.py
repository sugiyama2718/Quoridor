# coding:utf-8
import State

num2str = {-1: "j", 0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f", 6:"g", 7:"h", 8:"i", 9:"j"}


class Agent(object):
    # color=1 if white
    def __init__(self, color):
        self.color = color
        self.prev_action = None  # 相手のaction

    def act(self, state, showNQ=False):
        pass


def move_id2dxdy(move_id):
    dx = move_id // 3
    dy = move_id % 3
    if dx == 2:
        dx = -1
    if dy == 2:
        dy = -1
    return dx, dy


def dxdy2actionid(dx, dy):
    dx = int(dx >= 1) - int(dx <= -1)
    dy = int(dy >= 1) - int(dy <= -1)
    dx2 = dx
    dy2 = dy
    if dx2 == -1:
        dx2 = 2
    if dy2 == -1:
        dy2 = 2
    return 128 + dx2 * 3 + dy2


def str2actionid(state, s):
    for action_id in range(128 + 9):
        action_str = actionid2str(state, action_id)
        if action_str == s:
            return action_id
    return -1


def actionid2str(state, action_id):
    x, y = state.color_p(state.turn % 2)
    id1 = action_id // ((State.BOARD_LEN - 1) * (State.BOARD_LEN - 1))
    id2 = action_id % ((State.BOARD_LEN - 1) * (State.BOARD_LEN - 1))
    if id1 <= 1:
        x2 = id2 // (State.BOARD_LEN - 1)
        y2 = id2 % (State.BOARD_LEN - 1)
        s = num2str[x2] + str(y2 + 1)
        if id1 == 0:
            s += "h"
        else:
            s += "v"
    else:
        dx, dy = move_id2dxdy(id2)
        x2 = x + dx
        y2 = y + dy
        if (state.Bx == x2 and state.By == y2) or (state.Wx == x2 and state.Wy == y2):
            x2 += dx
            y2 += dy
        s = num2str[x2] + str(y2 + 1)
    return s


def is_jump_move(state, action_id):
    x, y = state.color_p(state.turn % 2)
    id1 = action_id // ((State.BOARD_LEN - 1) * (State.BOARD_LEN - 1))
    id2 = action_id % ((State.BOARD_LEN - 1) * (State.BOARD_LEN - 1))
    if id1 == 2:
        dx, dy = move_id2dxdy(id2)
        x2 = x + dx
        y2 = y + dy
        if (state.Bx == x2 and state.By == y2) or (state.Wx == x2 and state.Wy == y2):
            return True
    return False

