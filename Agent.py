# coding:utf-8
import State
from State import color_p

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

def str2actionid_statevec(statevec, s):
    for action_id in range(128 + 9):
        action_str = actionid2str_statevec(statevec, action_id)
        if action_str == s:
            return action_id
    return -1


def _actionid2str_helper(x, y, action_id, bx, by, wx, wy):
    """
    x, y      : 現在の操作対象の駒の位置 (int)
    action_id : アクションID (int)
    bx, by    : ボード上の黒駒の位置 (int)
    wx, wy    : ボード上の白駒の位置 (int)
    
    戻り値は action_id を文字列に変換したものです。
    """
    board_len = State.BOARD_LEN  # StateクラスのBOARD_LENにアクセス

    # (State.BOARD_LEN-1) * (State.BOARD_LEN-1) を計算
    sub_actions = (board_len - 1) * (board_len - 1)
    
    id1 = action_id // sub_actions
    id2 = action_id % sub_actions

    if id1 <= 1:
        # id1が0または1なら、壁を置く操作とみなす
        x2 = id2 // (board_len - 1)
        y2 = id2 % (board_len - 1)
        # num2str は盤上のx座標を文字に変換する辞書などと仮定
        s = num2str[x2] + str(y2 + 1)
        if id1 == 0:
            s += "h"
        else:
            s += "v"
    else:
        # id1が2以上なら、駒を動かす操作とみなす
        dx, dy = move_id2dxdy(id2)
        x2 = x + dx
        y2 = y + dy
        # 動かす先に既にどちらかの駒があるなら，さらに同じ方向へ1マス進める
        if (bx == x2 and by == y2) or (wx == x2 and wy == y2):
            x2 += dx
            y2 += dy
        s = num2str[x2] + str(y2 + 1)
    return s


def actionid2str(state, action_id):
    """
    既存の state を受け取るバージョン。
    stateから現在の駒の位置は color_p(state, state.turn % 2) により求める。
    """
    # ここでは color_p の動作として，
    # state.turn % 2 が 0 なら黒駒の位置, 1 なら白駒の位置を返すとする
    if state.turn % 2 == 0:
        x, y = state.Bx, state.By
    else:
        x, y = state.Wx, state.Wy

    return _actionid2str_helper(x, y, action_id, state.Bx, state.By, state.Wx, state.Wy)


def actionid2str_statevec(statevec, action_id):
    """
    statevecを受け取るバージョン。
    statevec は get_state_vec(state) して得られる tuple とする。
    
    statevec の各要素の意味は以下の通り:
      index 0 : state.Bx
      index 1 : state.By
      index 2 : state.Wx
      index 3 : state.Wy
      index 6 : state.turn % 2
      
    以上より、turn==0 なら (Bx, By), turn==1 なら (Wx, Wy) を現在の駒の位置とする。
    """
    # statevecはtupleなのでインデックス指定でアクセス可能
    bx = statevec[0]
    by = statevec[1]
    wx = statevec[2]
    wy = statevec[3]
    turn = statevec[6]
    
    if turn == 0:
        x, y = bx, by
    else:
        x, y = wx, wy

    return _actionid2str_helper(x, y, action_id, bx, by, wx, wy)


def is_jump_move(state, action_id):
    x, y = color_p(state, state.turn % 2)
    id1 = action_id // ((State.BOARD_LEN - 1) * (State.BOARD_LEN - 1))
    id2 = action_id % ((State.BOARD_LEN - 1) * (State.BOARD_LEN - 1))
    if id1 == 2:
        dx, dy = move_id2dxdy(id2)
        x2 = x + dx
        y2 = y + dy
        if (state.Bx == x2 and state.By == y2) or (state.Wx == x2 and state.Wy == y2):
            return True
    return False

