from sys import exit
# coding:utf-8
#cython: language_level=3, boundscheck=False
# cython: profile=True
import numpy as np
cimport numpy as np
import sys, os
import time
import copy
import collections
import math
from bitarray import bitarray

DTYPE = np.int32
ctypedef np.int32_t DTYPE_t
ctypedef np.float32_t DTYPE_float

BOARD_LEN = 9
DRAW_TURN = 300
CHANNEL = 15
CALC_DIST_ARRAY = True
notation_dict = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8}
total_time1 = 0.
total_time2 = 0.
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
BITARRAY_SIZE = 64


def select_action(DTYPE_float[:] Q, DTYPE_float[:] N, DTYPE_float[:] P, float C_puct, use_estimated_V, float estimated_V, color, turn, use_average_Q):
    if use_estimated_V:
        pass
    else:
        print("not implemented")
        exit()

    N_sum = 0
    for i in range(N.shape[0]):
        N_sum += N[i]
    #N_sum = sum(N)
    N_sum_sqrt = math.sqrt(1 + N_sum)
    a = -1
    x_max = -2.0
    for i in range(Q.shape[0]):
        if P[i] == 0.0:
            continue

        Ni = N[i]
        if Ni == 0:
            Qi = estimated_V
        else:
            Qi = Q[i]

        if color == turn % 2:
            x = Qi
        else:
            x = -Qi

        x = x + C_puct * P[i] * N_sum_sqrt / (1 + Ni)
        if x > x_max:
            a = i
            x_max = x
    return a


class Edge:
    def __init__(self, n, p, type_):
        self.n = n
        self.p = p
        self.type = type_

    #def __eq__(self, edge):
    #    return self.n == edge.n

    def __repr__(self):
        return "{} {} {}".format(self.n, self.p, self.type)


cdef class State:
    draw_turn = DRAW_TURN
    cdef public np.ndarray seen, row_wall, column_wall, must_be_checked_x, must_be_checked_y, placable_r_, placable_c_, placable_rb, placable_cb, placable_rw, placable_cw, row_special_cut, row_eq, column_special_cut, column_eq, dist_array1, dist_array2
    cdef public int Bx, By, Wx, Wy, turn, black_walls, white_walls, terminate, reward, wall0_terminate, pseudo_terminate, pseudo_reward
    cdef public row_graph, column_graph
    cdef public DTYPE_t[:, :, :] prev, cross_movable_arr
    cdef public row_wall_bit, column_wall_bit
    def __init__(self):
        self.row_wall = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.column_wall = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.row_wall_bit = bitarray(BITARRAY_SIZE)
        self.column_wall_bit = bitarray(BITARRAY_SIZE)
        self.placable_r_ = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.placable_c_ = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.placable_rb = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.placable_cb = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.placable_rw = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.placable_cw = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.must_be_checked_x = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.must_be_checked_y = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.Bx = BOARD_LEN // 2
        self.By = BOARD_LEN - 1
        self.Wx = BOARD_LEN // 2
        self.Wy = 0
        self.turn = 0
        self.black_walls = 10
        self.white_walls = 10

        self.terminate = False
        self.wall0_terminate = False  # 壁0を葉ノード扱いするときに使用 こちらは壁0で自動でTrueになる
        self.reward = 0  # blackから見たreward
        self.pseudo_terminate = False  # 相手の壁が0でこちらの方が2近い等、勝敗が確定しているときにTrue
        self.pseudo_reward = 0

        self.seen = np.zeros((BOARD_LEN, BOARD_LEN), dtype="bool")
        self.row_special_cut = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="int8")
        self.row_eq = np.zeros((BOARD_LEN, BOARD_LEN), dtype="int8")
        self.row_graph = {0: [[], {range(81)}, True, True]}
        self.column_special_cut = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="int8")
        self.column_eq = np.zeros((BOARD_LEN, BOARD_LEN), dtype="int8")
        self.column_graph = {0: [[], {range(81)}, True, True]}
        self.dist_array1 = np.zeros((BOARD_LEN, BOARD_LEN), dtype="int8")
        self.dist_array2 = np.zeros((BOARD_LEN, BOARD_LEN), dtype="int8")

        for y in range(BOARD_LEN):
            self.dist_array1[:, y] = y
            self.dist_array2[:, y] = BOARD_LEN - 1 - y

        self.cross_movable_arr = np.ones((BOARD_LEN, BOARD_LEN, 4), dtype="int32")
        for x in range(BOARD_LEN):
            self.cross_movable_arr[x, 0, UP] = 0
        for y in range(BOARD_LEN):
            self.cross_movable_arr[BOARD_LEN - 1, y, RIGHT] = 0
        for x in range(BOARD_LEN):
            self.cross_movable_arr[x, BOARD_LEN - 1, DOWN] = 0
        for y in range(BOARD_LEN):
            self.cross_movable_arr[0, y, LEFT] = 0

    def __eq__(self, state):
        f = np.all(self.row_wall == state.row_wall) and np.all(self.column_wall == state.column_wall)
        f = f and self.Bx == state.Bx and self.By == state.By and self.Wx == state.Wx and self.Wy == state.Wy
        f = f and self.black_walls == state.black_walls and self.white_walls == state.white_walls
        return f

    def get_player_dist_from_goal(self):
        return self.dist_array1[self.Bx, self.By], self.dist_array2[self.Wx, self.Wy]

    def color_p(self, color):
        if color == 0:
            return self.Bx, self.By
        else:
            return self.Wx, self.Wy

    def accept_action_str(self, s, check_placable=True, calc_placable_array=True, check_movable=True):
        # calc_placable_array=Falseにした場合は、以降正しく壁のおける場所を求められないことに注意
        #cdef np.ndarray[DTYPE_t, ndim = 3] cross_arr
        if len(s) <= 1 or len(s) >= 4:
            return False
        if s[0] not in notation_dict.keys():
            return False
        if not s[1].isdigit():
            return False
        x = notation_dict[s[0]]
        y = int(s[1]) - 1
        if len(s) == 2:
            if self.turn % 2 == 0:
                x2 = self.Bx
                y2 = self.By
            else:
                x2 = self.Wx
                y2 = self.Wy
            dx = x - x2
            dy = y - y2
            if abs(dx) + abs(dy) >= 3:
                return False
            if abs(dx) == 2 or abs(dy) == 2:
                x3 = x2 + dx // 2
                y3 = y2 + dy // 2
                if not ((self.Bx == x3 and self.By == y3) or (self.Wx == x3 and self.Wy == y3)):
                    return False
                dx //= 2
                dy //= 2
            if check_movable:
                mv = self.movable_array(x2, y2)
                if not mv[dx, dy]:
                    return False
            if self.turn % 2 == 0:
                self.Bx = x
                self.By = y
            else:
                self.Wx = x
                self.Wy = y

            # placable_arrayを更新
            if calc_placable_array:
                placable_r, placable_c = self.calc_placable_array(skip_calc_graph=True)
                self.placable_r_ = placable_r
                self.placable_c_ = placable_c
                self.placable_rb, self.placable_cb = (placable_r * (self.black_walls >= 1), placable_c * self.black_walls >= 1)
                self.placable_rw, self.placable_cw = (placable_r * (self.white_walls >= 1), placable_c * self.white_walls >= 1)

        elif len(s) == 3:
            if self.turn % 2 == 0:
                walls = self.black_walls
            else:
                walls = self.white_walls

            if check_placable:
                #rf, cf = self.placable(x, y)
                rf = self.placable_r_[x, y]  # 240121修正。こうしないとGUI.pyなどで道を塞げてしまう。むしろこれで問題ないなら速度の観点からもこちらに統一すべき
                cf = self.placable_c_[x, y]
                #print(rf, cf)

                if s[2] == "h":
                    if rf and walls >= 1:
                        self.row_wall[x, y] = 1
                        self.row_wall_bit[x * (BOARD_LEN - 1) + y] = 1
                        if self.turn % 2 == 0:
                            self.black_walls -= 1
                        else:
                            self.white_walls -= 1
                    else:
                        return False
                elif s[2] == "v":
                    if cf and walls >= 1:
                        self.column_wall[x, y] = 1
                        self.column_wall_bit[x * (BOARD_LEN - 1) + y] = 1
                        if self.turn % 2 == 0:
                            self.black_walls -= 1
                        else:
                            self.white_walls -= 1
                    else:
                        return False
                else:
                    return False
            else:  # 非合法手が来ない前提で、placableを省略して高速化
                if s[2] == "h":
                    if walls >= 1:
                        self.row_wall[x, y] = 1
                        self.row_wall_bit[x * (BOARD_LEN - 1) + y] = 1
                        if self.turn % 2 == 0:
                            self.black_walls -= 1
                        else:
                            self.white_walls -= 1
                    else:
                        return False
                elif s[2] == "v":
                    if walls >= 1:
                        self.column_wall[x, y] = 1
                        self.column_wall_bit[x * (BOARD_LEN - 1) + y] = 1
                        if self.turn % 2 == 0:
                            self.black_walls -= 1
                        else:
                            self.white_walls -= 1
                    else:
                        return False
                else:
                    return False

            # 壁置きの場合計算しなおし
            self.cross_movable_arr = self.cross_movable_array_by_diff(self.cross_movable_arr, x, y, s[2] == "h")
            if calc_placable_array:
                placable_r, placable_c = self.calc_placable_array()
                self.placable_r_ = placable_r
                self.placable_c_ = placable_c
                self.placable_rb, self.placable_cb = (placable_r * (self.black_walls >= 1), placable_c * self.black_walls >= 1)
                self.placable_rw, self.placable_cw = (placable_r * (self.white_walls >= 1), placable_c * self.white_walls >= 1)
            
            # dist_arrayも計算しなおし
            if CALC_DIST_ARRAY:
                self.dist_array1 = self.dist_array(0, self.cross_movable_arr)
                self.dist_array2 = self.dist_array(BOARD_LEN - 1, self.cross_movable_arr)
        self.turn += 1

        if self.By == 0:
            self.terminate = True
            self.reward = 1
        elif self.Wy == BOARD_LEN - 1:
            self.terminate = True
            self.reward = -1
        elif self.turn == self.draw_turn:
            self.terminate = True
            self.reward = 0

        if (self.black_walls == 0 and self.white_walls == 0) or self.terminate:
            self.wall0_terminate = True
        
        if self.terminate:
            self.pseudo_terminate = True
            self.pseudo_reward = self.reward
        else:
            B_dist = self.dist_array1[self.Bx, self.By]
            W_dist = self.dist_array2[self.Wx, self.Wy]
            if self.black_walls == 0 and (W_dist + (1 - self.turn % 2) <= B_dist - 1):
                self.pseudo_terminate = True
                self.pseudo_reward = -1
            elif self.white_walls == 0 and (B_dist + self.turn % 2 <= W_dist - 1):
                self.pseudo_terminate = True
                self.pseudo_reward = 1
            elif self.is_mirror_match():
                self.pseudo_terminate = True
                self.pseudo_reward = -1
            else:
                self.pseudo_terminate = False
                self.pseudo_reward = 0

        return True

    def is_mirror_match(self):
        # 盤面上の壁が5枚以下ではmirror matchは成立し得ない
        if 20 - (self.black_walls + self.white_walls) <= 5:
            return False
        
        if self.black_walls != self.white_walls:
            return False

        # 壁が回転対称でなければ
        if not (np.all(self.row_wall == np.flip(self.row_wall)) and np.all(self.column_wall == np.flip(self.column_wall))):
            return False
        
        # 中央マスから横に移動できる場合、先手は横に移動することで優位に立てる可能性がある
        if not (self.column_wall[3, 3] or self.column_wall[4, 3] or self.row_wall[3, 3] or self.row_wall[4, 3]):
            return False
        
        # コマが回転対称で飛び越し前かつ先手番なら後手勝ち
        f1 = (self.Bx == 8 - self.Wx and self.By == 8 - self.Wy and self.turn % 2 == 0 and self.dist_array1[self.Bx, self.By] > self.dist_array1[4, 4])

        # 飛び越し後は逆
        f2 = (self.Bx == 8 - self.Wx and self.By == 8 - self.Wy and self.turn % 2 == 1 and self.dist_array1[self.Bx, self.By] < self.dist_array1[4, 4])
        
        if not (f1 or f2):
            return False
        
        # ゴールへの道が中央マスを必ず通る場合のみ後手勝利。中央マスの上側を塞いだとき、ゴールにたどり着けなくなるかどうかで判定。
        blocked_cross_movable_array = np.copy(self.cross_movable_arr)
        if self.column_wall[3, 3] or self.column_wall[4, 3]:
            blocked_cross_movable_array[4, 3, DOWN] = 0
            blocked_cross_movable_array[4, 4, UP] = 0
            blocked_cross_movable_array[4, 4, DOWN] = 0
            blocked_cross_movable_array[4, 5, UP] = 0
            blocked_dist_array = self.dist_array(0, blocked_cross_movable_array)
            if blocked_dist_array[4, 3] != np.max(blocked_dist_array) and blocked_dist_array[4, 5] != np.max(blocked_dist_array):
                return False
        else:
            blocked_cross_movable_array[3, 4, RIGHT] = 0
            blocked_cross_movable_array[4, 4, LEFT] = 0
            blocked_cross_movable_array[4, 4, RIGHT] = 0
            blocked_cross_movable_array[5, 4, LEFT] = 0
            blocked_dist_array = self.dist_array(0, blocked_cross_movable_array)
            if blocked_dist_array[3, 4] != np.max(blocked_dist_array) and blocked_dist_array[5, 4] != np.max(blocked_dist_array):
                return False

        return True

    def is_certain_path_terminate(self, color=None):
        B_dist = self.dist_array1[self.Bx, self.By]
        W_dist = self.dist_array2[self.Wx, self.Wy]

        # 先手後手それぞれで確定路があるか調べる
        if (color is None or color == 0) and B_dist + self.turn % 2 <= W_dist - 1:  #  B_dist <= B_certain_distなのでこれを満たさないときは判定不要
            placable_r, placable_c = self.calc_oneside_placable_cand_from_color(0)
            certain_cross_movable_arr = self.cross_movable_array2(self.row_wall | placable_r, self.column_wall | placable_c)
            B_certain_dist = self.shortest_path_len(self.Bx, self.By, 0, certain_cross_movable_arr)
            if B_certain_dist + self.turn % 2 <= W_dist - 1:
                return True

        if (color is None or color == 1) and W_dist + (1 - self.turn % 2) <= B_dist - 1:
            placable_r, placable_c = self.calc_oneside_placable_cand_from_color(1)
            certain_cross_movable_arr = self.cross_movable_array2(self.row_wall | placable_r, self.column_wall | placable_c)
            W_certain_dist = self.shortest_path_len(self.Wx, self.Wy, BOARD_LEN - 1, certain_cross_movable_arr)
            if W_certain_dist + (1 - self.turn % 2) <= B_dist - 1:
                return True

        return False

    # 上，右，下，左
    def cross_movable(self, x, y):
        ret = np.zeros((4,), dtype="bool")
        ret[UP] = (not (y == 0 or self.row_wall[min(x, BOARD_LEN - 2), y - 1] or self.row_wall[max(x - 1, 0), y - 1]))
        ret[RIGHT] = (not (x == BOARD_LEN - 1 or self.column_wall[x, min(y, BOARD_LEN - 2)] or self.column_wall[x, max(y - 1, 0)]))
        ret[DOWN] = (not (y == BOARD_LEN - 1 or self.row_wall[min(x, BOARD_LEN - 2), y] or self.row_wall[max(x - 1, 0), y]))
        ret[LEFT] = (not (x == 0 or self.column_wall[x - 1, min(y, BOARD_LEN - 2)] or self.column_wall[x - 1, max(y - 1, 0)]))
        return ret

    # def cross_movable_array(self):
    #     if self.cross_movable_arr is None:
    #         self.cross_movable_arr = self.cross_movable_array2(self.row_wall, self.column_wall)
    #     return np.copy(self.cross_movable_arr)

    def set_state_by_wall(self):
        # row_wallなどを直接指定して状態を作るときに用いる
        # 差分計算している部分を計算する

        self.cross_movable_arr = self.cross_movable_array2(self.row_wall, self.column_wall)
        self.dist_array1 = self.dist_array(0, self.cross_movable_arr)
        self.dist_array2 = self.dist_array(BOARD_LEN - 1, self.cross_movable_arr)

        placable_r, placable_c = self.calc_placable_array()
        self.placable_r_ = placable_r
        self.placable_c_ = placable_c
        self.placable_rb, self.placable_cb = (placable_r * (self.black_walls >= 1), placable_c * self.black_walls >= 1)
        self.placable_rw, self.placable_cw = (placable_r * (self.white_walls >= 1), placable_c * self.white_walls >= 1)

    def cross_movable_array_by_diff(self, DTYPE_t[:, :, :] prev_cross_movable_arr, int wall_x, int wall_y, int is_row):
        if is_row:
            for x in range(wall_x, wall_x + 2):
                prev_cross_movable_arr[x, wall_y, DOWN] = 0
                prev_cross_movable_arr[x, wall_y+1, UP] = 0
        else:
            for y in range(wall_y, wall_y + 2):
                prev_cross_movable_arr[wall_x, y, RIGHT] = 0
                prev_cross_movable_arr[wall_x+1, y, LEFT] = 0
        return prev_cross_movable_arr

    def cross_movable_array2(self, row_wall, column_wall):
        cdef int x, y
        #cdef np.ndarray[DTYPE_t, ndim = 3] ret = np.zeros((BOARD_LEN, BOARD_LEN, 4), dtype=DTYPE)
        cdef DTYPE_t[:, :, :] ret = np.zeros((BOARD_LEN, BOARD_LEN, 4), dtype=DTYPE)

        for x in range(BOARD_LEN):
            for y in range(BOARD_LEN):
                ret[x, y, 0] = (not (y == 0 or row_wall[min(x, BOARD_LEN - 2), y - 1] or row_wall[max(x - 1, 0), y - 1]))
                ret[x, y, 1] = (not (x == BOARD_LEN - 1 or column_wall[x, min(y, BOARD_LEN - 2)] or column_wall[x, max(y - 1, 0)]))
                ret[x, y, 2] = (not (y == BOARD_LEN - 1 or row_wall[min(x, BOARD_LEN - 2), y] or row_wall[max(x - 1, 0), y]))
                ret[x, y, 3] = (not (x == 0 or column_wall[x - 1, min(y, BOARD_LEN - 2)] or column_wall[x - 1, max(y - 1, 0)]))

        # ret[:, 0, UP] = 1
        # ret[:-1, 1:, UP] += row_wall  # yはかぶらない
        # ret[1:, 1:, UP] += row_wall  # x方向には連続して壁は置かれない
        # ret[:, :, UP] = 1 - ret[:, :, UP]

        # ret[-1, :, RIGHT] = 1
        # ret[:-1, :-1, RIGHT] += column_wall 
        # ret[:-1, 1:, RIGHT] += column_wall 
        # ret[:, :, RIGHT] = 1 - ret[:, :, RIGHT]

        # ret[:, -1, DOWN] = 1
        # ret[:-1, :-1, DOWN] += row_wall 
        # ret[1:, :-1, DOWN] += row_wall 
        # ret[:, :, DOWN] = 1 - ret[:, :, DOWN]

        # ret[0, :, LEFT] = 1
        # ret[1:, :-1, LEFT] += column_wall 
        # ret[1:, 1:, LEFT] += column_wall 
        # ret[:, :, LEFT] = 1 - ret[:, :, LEFT]

        return ret

    # shortest_onely=Trueの場合ゴールからの距離を縮める方向のみに1を立てる
    def movable_array(self, x, y, shortest_only=False):
        mv = np.zeros((3, 3), dtype="bool")
        cross = self.cross_movable(x, y)
        if shortest_only:
            if CALC_DIST_ARRAY:
                if self.turn % 2 == 0:
                    dist_arr = self.dist_array1
                else:
                    dist_arr = self.dist_array2
            else:
                dist_arr = self.dist_array(0 if self.turn % 2 == 0 else BOARD_LEN - 1, self.cross_movable_arr)
        for i, p in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            if not cross[i]:
                continue
            x2 = x + p[0]
            y2 = y + p[1]
            if (self.Bx == x2 and self.By == y2) or (self.Wx == x2 and self.Wy == y2):
                cross2 = self.cross_movable(x2, y2)

                # 同じ方向に進むことができるかからチェック。進めるなら斜めには移動できない。
                if cross2[i]:
                    if shortest_only:
                        if dist_arr[x2 + p[0], y2 + p[1]] < dist_arr[x, y]:
                            mv[p] = 1
                    else:
                        mv[p] = 1
                    continue

                movable_list = []
                for j, q in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
                    if not cross2[j]:
                        continue
                    if shortest_only and (dist_arr[x2 + q[0], y2 + q[1]] >= dist_arr[x, y]):
                        continue
                    movable_list.append((max(min(p[0] + q[0], 1), -1), max(min(p[1] + q[1], 1), -1)))
                for movable_p in movable_list:
                    mv[movable_p] = 1
                mv[0, 0] = 0
            else:
                if shortest_only:
                    if dist_arr[x2, y2] < dist_arr[x, y]:
                        mv[p] = 1
                else:
                    mv[p] = 1

        return mv

    def inboard(self, x, y, size):
        if x < 0 or y < 0 or x >= size or y >= size:
            return False
        return True

    def placable_array(self, color):
        if color == 0:
            return self.placable_rb, self.placable_cb
        else:
            return self.placable_rw, self.placable_cw

    def old_calc_placable_array(self):
        ret1 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        ret2 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        must_be_checked_x1 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        must_be_checked_x2 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        must_be_checked_y1 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        must_be_checked_y2 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        routes = [self.getroute(self.Bx, self.By, 0, True), self.getroute(self.Bx, self.By, 0, False)]

        pointss0r = []
        pointss0l = []
        pointss1d = []
        pointss1u = []
        for route in routes:
            points0r = []
            points0l = []
            points1d = []
            points1u = []
            for i, p in enumerate(route):
                if i == len(route) - 1:
                    break
                dx = route[i + 1][0] - route[i][0]
                dy = route[i + 1][1] - route[i][1]
                dx = (dx + 2) // 2 - 1
                dy = (dy + 2) // 2 - 1
                x_move = abs(route[i + 1][0] - route[i][0]) >= 1
                y_move = abs(route[i + 1][1] - route[i][1]) >= 1
                if self.inboard(p[0] + dx, p[1] + dy, BOARD_LEN - 1):
                    if x_move and dx == 0:
                        points0r.append((p[0] + dx, p[1] + dy))
                    elif x_move and dx == -1:
                        points0l.append((p[0] + dx, p[1] + dy))
                    elif y_move and dy == 0:
                        points1d.append((p[0] + dx, p[1] + dy))
                    elif y_move and dy == -1:
                        points1u.append((p[0] + dx, p[1] + dy))
                if x_move and dx == 0 and self.inboard(p[0] + dx, p[1] + dy - 1, BOARD_LEN - 1):
                    points0r.append((p[0] + dx, p[1] + dy - 1))
                elif x_move and dx == -1 and self.inboard(p[0] + dx, p[1] + dy - 1, BOARD_LEN - 1):
                    points0l.append((p[0] + dx, p[1] + dy - 1))
                elif y_move and dy == 0 and self.inboard(p[0] + dx - 1, p[1] + dy, BOARD_LEN - 1):
                    points1d.append((p[0] + dx - 1, p[1] + dy))
                elif y_move and dy == -1 and self.inboard(p[0] + dx - 1, p[1] + dy, BOARD_LEN - 1):
                    points1u.append((p[0] + dx - 1, p[1] + dy))
            pointss0r.append(points0r)
            pointss0l.append(points0l)
            pointss1d.append(points1d)
            pointss1u.append(points1u)
        and_points0r = pointss0r[0]
        and_points0l = pointss0l[0]
        and_points1d = pointss1d[0]
        and_points1u = pointss1u[0]
        for points in pointss0r[1:]:
            and_points0r = list(set(and_points0r) & set(points))
        for points in pointss0l[1:]:
            and_points0l = list(set(and_points0l) & set(points))
        for points in pointss1d[1:]:
            and_points1d = list(set(and_points1d) & set(points))
        for points in pointss1u[1:]:
            and_points1u = list(set(and_points1u) & set(points))
        for x, y in and_points0r + and_points0l:
            must_be_checked_x1[x, y] = True
        for x, y in and_points1d + and_points1u:
            must_be_checked_y1[x, y] = True

        routes = [self.getroute(self.Wx, self.Wy, BOARD_LEN - 1, True), self.getroute(self.Wx, self.Wy, BOARD_LEN - 1, False)]

        pointss0r = []
        pointss0l = []
        pointss1d = []
        pointss1u = []
        for route in routes:
            points0r = []
            points0l = []
            points1d = []
            points1u = []
            for i, p in enumerate(route):
                if i == len(route) - 1:
                    break
                dx = route[i + 1][0] - route[i][0]
                dy = route[i + 1][1] - route[i][1]
                dx = (dx + 2) // 2 - 1
                dy = (dy + 2) // 2 - 1
                x_move = abs(route[i + 1][0] - route[i][0]) >= 1
                y_move = abs(route[i + 1][1] - route[i][1]) >= 1
                if self.inboard(p[0] + dx, p[1] + dy, BOARD_LEN - 1):
                    if x_move and dx == 0:
                        points0r.append((p[0] + dx, p[1] + dy))
                    elif x_move and dx == -1:
                        points0l.append((p[0] + dx, p[1] + dy))
                    elif y_move and dy == 0:
                        points1d.append((p[0] + dx, p[1] + dy))
                    elif y_move and dy == -1:
                        points1u.append((p[0] + dx, p[1] + dy))
                if x_move and dx == 0 and self.inboard(p[0] + dx, p[1] + dy - 1, BOARD_LEN - 1):
                    points0r.append((p[0] + dx, p[1] + dy - 1))
                elif x_move and dx == -1 and self.inboard(p[0] + dx, p[1] + dy - 1, BOARD_LEN - 1):
                    points0l.append((p[0] + dx, p[1] + dy - 1))
                elif y_move and dy == 0 and self.inboard(p[0] + dx - 1, p[1] + dy, BOARD_LEN - 1):
                    points1d.append((p[0] + dx - 1, p[1] + dy))
                elif y_move and dy == -1 and self.inboard(p[0] + dx - 1, p[1] + dy, BOARD_LEN - 1):
                    points1u.append((p[0] + dx - 1, p[1] + dy))
            pointss0r.append(points0r)
            pointss0l.append(points0l)
            pointss1d.append(points1d)
            pointss1u.append(points1u)
        and_points0r = pointss0r[0]
        and_points0l = pointss0l[0]
        and_points1d = pointss1d[0]
        and_points1u = pointss1u[0]
        for points in pointss0r[1:]:
            and_points0r = list(set(and_points0r) & set(points))
        for points in pointss0l[1:]:
            and_points0l = list(set(and_points0l) & set(points))
        for points in pointss1d[1:]:
            and_points1d = list(set(and_points1d) & set(points))
        for points in pointss1u[1:]:
            and_points1u = list(set(and_points1u) & set(points))
        for x, y in and_points0r + and_points0l:
            must_be_checked_x2[x, y] = True
        for x, y in and_points1d + and_points1u:
            must_be_checked_y2[x, y] = True

        self.must_be_checked_x = must_be_checked_x1 | must_be_checked_x2
        self.must_be_checked_y = must_be_checked_y1 | must_be_checked_y2

        for x in range(BOARD_LEN - 1):
            for y in range(BOARD_LEN - 1):
                f1, f2 = self.placable(x, y)
                ret1[x, y] = f1
                ret2[x, y] = f2

        return ret1, ret2

    cdef (int, int) placable_with_color(self, x, y, color):
        # すべてarrivableで確かめる。
        if self.row_wall[x, y] or self.column_wall[x, y]:
            return False, False
        row_f = True
        column_f = True

        if self.row_wall[max(x - 1, 0), y] or self.row_wall[min(x + 1, BOARD_LEN - 2), y]:
            row_f = False
        if self.column_wall[x, max(y - 1, 0)] or self.column_wall[x, min(y + 1, BOARD_LEN - 2)]:
            column_f = False
        if row_f:
            self.row_wall[x, y] = 1
            self.row_wall_bit[x * (BOARD_LEN - 1) + y] = 1
            if color == 0:
                f = self.arrivable(self.Bx, self.By, 0) 
            else:
                f = self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.row_wall[x, y] = 0
            self.row_wall_bit[x * (BOARD_LEN - 1) + y] = 0
            row_f = row_f and f
        if column_f:
            self.column_wall[x, y] = 1
            self.column_wall_bit[x * (BOARD_LEN - 1) + y] = 1
            if color == 0:
                f = self.arrivable(self.Bx, self.By, 0)
            else:
                f = self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.column_wall[x, y] = 0
            self.column_wall_bit[x * (BOARD_LEN - 1) + y] = 0
            column_f = column_f and f
        return row_f, column_f
    
    cdef (int, int) placable(self, x, y):
        if self.row_wall[x, y] or self.column_wall[x, y]:
            return False, False
        row_f = True
        column_f = True
    
        # if x == 2 and y == 2 and self.turn == 17:
        #     print(self.row_wall)
        #     print(self.column_wall)
        #     print()

        if self.row_wall[max(x - 1, 0), y] or self.row_wall[min(x + 1, BOARD_LEN - 2), y]:
            row_f = False
        if self.column_wall[x, max(y - 1, 0)] or self.column_wall[x, min(y + 1, BOARD_LEN - 2)]:
            column_f = False
        if row_f and self.must_be_checked_y[x, y]:
            self.row_wall[x, y] = 1
            self.row_wall_bit[x * (BOARD_LEN - 1) + y] = 1
            f = self.arrivable(self.Bx, self.By, 0) and self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.row_wall[x, y] = 0
            self.row_wall_bit[x * (BOARD_LEN - 1) + y] = 0
            row_f = row_f and f
        if column_f and self.must_be_checked_x[x, y]:
            self.column_wall[x, y] = 1
            self.column_wall_bit[x * (BOARD_LEN - 1) + y] = 1
            f = self.arrivable(self.Bx, self.By, 0) and self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.column_wall[x, y] = 0
            self.column_wall_bit[x * (BOARD_LEN - 1) + y] = 0
            column_f = column_f and f
        return row_f, column_f

    def placable_r(self, x, y):
        if self.row_wall[x, y]:
            return False
        row_f = True

        if self.row_wall[max(x - 1, 0), y] or self.row_wall[min(x + 1, BOARD_LEN - 2), y]:
            row_f = False
        if row_f and self.must_be_checked_y[x, y]:
            self.row_wall[x, y] = 1
            self.row_wall_bit[x * (BOARD_LEN - 1) + y] = 1
            f = self.arrivable(self.Bx, self.By, 0) and self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.row_wall[x, y] = 0
            self.row_wall_bit[x * (BOARD_LEN - 1) + y] = 0
            row_f = row_f and f
        return row_f

    def placable_c(self, x, y):
        if self.column_wall[x, y]:
            return False
        column_f = True

        if self.column_wall[x, max(y - 1, 0)] or self.column_wall[x, min(y + 1, BOARD_LEN - 2)]:
            column_f = False
        if column_f and self.must_be_checked_x[x, y]:
            self.column_wall[x, y] = 1
            self.column_wall_bit[x * (BOARD_LEN - 1) + y] = 1
            f = self.arrivable(self.Bx, self.By, 0) and self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.column_wall[x, y] = 0
            self.column_wall_bit[x * (BOARD_LEN - 1) + y] = 0
            column_f = column_f and f
        return column_f

    def shortest(self, x, y, goal_y):
        dist = np.ones((BOARD_LEN, BOARD_LEN)) * BOARD_LEN * BOARD_LEN * 2
        prev = np.zeros((BOARD_LEN, BOARD_LEN, 2), dtype="int8")
        seen = np.zeros((BOARD_LEN, BOARD_LEN))
        dist[x, y] = 0
        while np.sum(seen) < BOARD_LEN * BOARD_LEN:
            x2, y2 = np.unravel_index(np.argmin(dist + seen * BOARD_LEN * BOARD_LEN * 3, axis=None), dist.shape)
            seen[x2, y2] = 1
            cross = self.cross_movable(x2, y2)
            for i, p in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
                x3 = x2 + p[0]
                y3 = y2 + p[1]
                if cross[i]:
                    if dist[x3, y3] > dist[x2, y2] + 1:
                        dist[x3, y3] = dist[x2, y2] + 1
                        prev[x3, y3, 0] = x2
                        prev[x3, y3, 1] = y2
        x4 = np.argmin(dist[:, goal_y])
        y4 = goal_y
        route = []
        while not (x4 == x and y4 == y):
            route.append((x4, y4))
            x4, y4 = prev[x4, y4]
        route.append((x, y))
        route = route[::-1]
        return route

    def shortest_path_len(self, x, y, goal_y, cross_arr):
        dist = np.ones((BOARD_LEN, BOARD_LEN)) * BOARD_LEN * BOARD_LEN * 2
        prev = np.zeros((BOARD_LEN, BOARD_LEN, 2), dtype="int8")
        seen = np.zeros((BOARD_LEN, BOARD_LEN))
        dist[x, y] = 0
        while np.sum(seen) < BOARD_LEN * BOARD_LEN:
            x2, y2 = np.unravel_index(np.argmin(dist + seen * BOARD_LEN * BOARD_LEN * 3, axis=None), dist.shape)
            seen[x2, y2] = 1
            for i, p in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
                x3 = x2 + p[0]
                y3 = y2 + p[1]
                if cross_arr[x2, y2, i]:
                    if dist[x3, y3] > dist[x2, y2] + 1:
                        dist[x3, y3] = dist[x2, y2] + 1
                        prev[x3, y3, 0] = x2
                        prev[x3, y3, 1] = y2
        return np.min(dist[:, goal_y])

    def dist_array(self, int goal_y, DTYPE_t[:, :, :] cross_arr):
        cdef int x, x2, y2, x3, y3, i, dx, dy
        cdef np.ndarray[DTYPE_t, ndim = 2] dist = np.ones((BOARD_LEN, BOARD_LEN), dtype=DTYPE) * BOARD_LEN * BOARD_LEN * 2

        queue = []
        for x in range(BOARD_LEN):
            dist[x, goal_y] = 0
            queue.append((x, goal_y))
        while len(queue) > 0:
            x2, y2 = queue.pop(0)
            for i, dx, dy in [(0, 0, -1), (1, 1, 0), (2, 0, 1), (3, -1, 0)]:
                x3 = x2 + dx
                y3 = y2 + dy
                if cross_arr[x2, y2, i] and dist[x3, y3] > dist[x2, y2] + 1:
                    dist[x3, y3] = dist[x2, y2] + 1
                    queue.append((x3, y3))
        dist[dist == BOARD_LEN * BOARD_LEN * 2] = -1
        dist[dist == -1] = np.max(dist) + 1
        return dist

    def arrivable(self, x, y, goal_y, isleft=True):
        self.seen = np.zeros((BOARD_LEN, BOARD_LEN), dtype="bool")
        self.prev = np.zeros((BOARD_LEN, BOARD_LEN, 2), dtype="int32")
        return self.arrivable_(x, y, goal_y, isleft)

    def getroute(self, x, y, goal_y, isleft):
        self.seen = np.zeros((BOARD_LEN, BOARD_LEN), dtype="bool")
        self.prev = np.zeros((BOARD_LEN, BOARD_LEN, 2), dtype="int32")
        self.arrivable_(x, y, goal_y, isleft)
        x2 = np.argmax(self.seen[:, goal_y])
        y2 = goal_y
        route = []
        while not (x2 == x and y2 == y):
            route.append((x2, y2))
            x2, y2 = self.prev[x2, y2]
        route.append((x, y))
        route = route[::-1]
        return route

    # この関数が返すedgeはEdgeクラスではなく(p, type)のタプルのリスト
    def get_blocking_edges(self, index, graph, is_goal_0):
        _, edgess = self.get_blocking_edges_rec(index, graph, is_goal_0, np.zeros((BOARD_LEN * BOARD_LEN), dtype="bool"), {})
        #print(edgess)
        n = len(edgess)
        part_edgess = []
        for x in edgess:
            part_edgess.append(list(set([(edge.p, edge.type) for edge in x])))
        part_edges = []
        for edges in part_edgess:
            part_edges.extend(edges)
        counter = collections.Counter(part_edges)
        ret_edges = []
        for edge, c in counter.items():
            if c >= n:
                ret_edges.append(edge)
        return ret_edges

    def get_blocking_edges_rec(self, index, graph, is_goal_0, seen, goal, prev_id=-1):
        seen[index] = True
        if is_goal_0 and graph[index][2]:
            goal[index] = []
            return True, []
        if not is_goal_0 and graph[index][3]:
            goal[index] = []
            return True, []

        ret_edgess = []
        ret_f = False
        for edge in graph[index][0]:
            if edge.n == prev_id:
                continue
            if edge.n in goal.keys():
                ret_f = True
                goal[index] = [edge] + goal[edge.n]
                ret_edgess.append([edge] + goal[edge.n])
                continue
            if seen[edge.n]:
                continue
            f, edgess = self.get_blocking_edges_rec(edge.n, graph, is_goal_0, seen, goal, index)
            ret_f = ret_f or f
            if f:
                goal[index] = [edge] + goal[edge.n]
                if len(edgess) == 0:
                    ret_edgess.append([edge])
                else:
                    ret_edgess.extend([[edge] + edges for edges in edgess])
        seen[index] = False
        return ret_f, ret_edgess
    
    def calc_oneside_placable_cand_from_color(self, color):
        # どちらかのプレイヤー（colorで指定）の路だけで壁置きの可能性を判定する。勝利の確定判定などに用いる。各種内部変数は計算済みと仮定する。

        cdef np.ndarray[DTYPE_t, ndim = 2] row_array = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 2] column_array = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        for x in range(BOARD_LEN - 1):
            for y in range(BOARD_LEN - 1):
                f1, f2 = self.placable_with_color(x, y, color)
                row_array[x, y] = f1
                column_array[x, y] = f2
        return row_array, column_array

    def cald_oneside_placable_cand(self, equivalence_class, graph, int x2, int y2, is_goal0):
        cdef int x, y
        edges = self.get_blocking_edges(equivalence_class[x2, y2], graph, is_goal0)

        # 実際置けない場所を求める
        cdef np.ndarray[DTYPE_t, ndim = 2] placable = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        for edge in edges:
            x, y = edge[0]
            if edge[1] == 1:
                if x > 0:
                    placable[x - 1, y] = 0
                if x < BOARD_LEN - 1:
                    placable[x, y] = 0
            if edge[1] == 2:
                placable[x, y] = 0
        return placable
    
    def calc_placable_array(self, skip_calc_graph=False):
        cdef np.ndarray[DTYPE_t, ndim = 2] row_array = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 2] column_array = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 2] wall_point_array = np.zeros((BOARD_LEN + 1, BOARD_LEN + 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 2] count_array_x = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 2] count_array_y = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)

        # 周囲を囲む
        wall_point_array[0, :] = 1
        wall_point_array[BOARD_LEN, :] = 1
        wall_point_array[:, 0] = 1
        wall_point_array[:, BOARD_LEN] = 1

        wall_point_array[:BOARD_LEN - 1, 1:BOARD_LEN] += np.array(self.row_wall, dtype=DTYPE)
        wall_point_array[1:BOARD_LEN, 1:BOARD_LEN] += np.array(self.row_wall, dtype=DTYPE)
        wall_point_array[2:, 1:BOARD_LEN] += np.array(self.row_wall, dtype=DTYPE)
        wall_point_array[1:BOARD_LEN, :BOARD_LEN - 1] += np.array(self.column_wall, dtype=DTYPE)
        wall_point_array[1:BOARD_LEN, 1:BOARD_LEN] += np.array(self.column_wall, dtype=DTYPE)
        wall_point_array[1:BOARD_LEN, 2:] += np.array(self.column_wall, dtype=DTYPE)

        wall_point_array = np.array(wall_point_array >= 1, dtype=DTYPE)

        count_array_x = wall_point_array[1:BOARD_LEN, :BOARD_LEN - 1] + wall_point_array[1:BOARD_LEN, 1:BOARD_LEN] + wall_point_array[1:BOARD_LEN, 2:]
        count_array_y = wall_point_array[:BOARD_LEN - 1, 1:BOARD_LEN] + wall_point_array[1:BOARD_LEN, 1:BOARD_LEN] + wall_point_array[2:, 1:BOARD_LEN]
        
        self.must_be_checked_x = (count_array_x >= 2)
        self.must_be_checked_y = (count_array_y >= 2)
        # if self.turn == 17:
        #     print(self.row_wall)
        #     print(self.column_wall)
        #     print(wall_point_array)
        #     print(count_array_x)
        #     print(self.must_be_checked_x)

        for x in range(BOARD_LEN - 1):
            for y in range(BOARD_LEN - 1):
                if row_array[x, y] and column_array[x, y]:
                    f1, f2 = self.placable(x, y)
                    row_array[x, y] = f1
                    column_array[x, y] = f2
                elif row_array[x, y]:  # columnはFalse
                    row_array[x, y] = self.placable_r(x, y)
                    column_array[x, y] = False
                elif column_array[x, y]:
                    row_array[x, y] = False
                    column_array[x, y] = self.placable_c(x, y)
                else:
                    row_array[x, y] = column_array[x, y] = False
        return row_array, column_array

    def old_calc_placable_array2(self, skip_calc_graph=False):
        cdef np.ndarray[DTYPE_t, ndim = 3] arr
        cdef np.ndarray[DTYPE_t, ndim = 2] row_array, column_array
        cdef DTYPE_t[:, :, :] cross_arr, cross_arr_copy
        cdef int x, y
        cross_arr = np.copy(self.cross_movable_arr)
        cross_arr_copy = np.copy(cross_arr)
        cdef np.ndarray[DTYPE_t, ndim = 2] row_cut1 = np.zeros((BOARD_LEN, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 2] row_cut2 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 2] column_cut1 = np.zeros((BOARD_LEN - 1, BOARD_LEN), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 2] column_cut2 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 2] column_cut1T, column_cut2T
        cdef np.ndarray[DTYPE_t, ndim = 3] cross_arr2 = np.zeros((BOARD_LEN, BOARD_LEN, 4), dtype=DTYPE)
        for arr in self.get_all_cycles(cross_arr):
            row_cut1 = row_cut1 | (arr[:, 1:, UP] & arr[:, :-1, DOWN])
            row_cut2 = row_cut2 | (arr[:-1, 1:, UP] & arr[1:, :-1, DOWN])
            column_cut1 = column_cut1 | (arr[:-1, :, RIGHT] & arr[1:, :, LEFT])
            column_cut2 = column_cut2 | (arr[:-1, :-1, RIGHT] & arr[1:, 1:, LEFT])

        if skip_calc_graph:
            row_array = self.placable_cand(self.row_eq, self.row_graph, self.Bx, self.By, self.Wx, self.Wy)
        else:
            row_array, self.row_special_cut, self.row_eq, self.row_graph = self.calc_placable_row_array(row_cut1, row_cut2, cross_arr, self.Bx, self.By, self.Wx, self.Wy, True)

        if skip_calc_graph:
            column_array = self.placable_cand(self.column_eq, self.column_graph, self.By, self.Bx, self.Wy, self.Wx)
            column_array = column_array.T
        else:
            cross_arr = cross_arr_copy
            for x in range(BOARD_LEN):
                for y in range(BOARD_LEN):
                    cross_arr2[y, x, LEFT] = cross_arr[x, y, UP]
                    cross_arr2[y, x, DOWN] = cross_arr[x, y, RIGHT]
                    cross_arr2[y, x, RIGHT] = cross_arr[x, y, DOWN]
                    cross_arr2[y, x, UP] = cross_arr[x, y, LEFT]
            # cross_arr2[:, :, LEFT] = cross_arr[:, :, UP].T
            # cross_arr2[:, :, DOWN] = cross_arr[:, :, RIGHT].T
            # cross_arr2[:, :, RIGHT] = cross_arr[:, :, DOWN].T
            # cross_arr2[:, :, UP] = cross_arr[:, :, LEFT].T
            column_cut1T = column_cut1.T
            column_cut2T = column_cut2.T
            column_array, self.column_special_cut, self.column_eq, self.column_graph = self.calc_placable_row_array(column_cut1T, column_cut2T, cross_arr2, self.By, self.Bx, self.Wy, self.Wx, False)
            column_array = column_array.T
            self.column_special_cut = self.column_special_cut.T

        self.must_be_checked_x = column_array & self.column_special_cut
        self.must_be_checked_y = row_array & self.row_special_cut

        for x in range(BOARD_LEN - 1):
            for y in range(BOARD_LEN - 1):
                if row_array[x, y] and column_array[x, y]:
                    f1, f2 = self.placable(x, y)
                    row_array[x, y] = f1
                    column_array[x, y] = f2
                elif row_array[x, y]:  # columnはFalse
                    row_array[x, y] = self.placable_r(x, y)
                    column_array[x, y] = False
                elif column_array[x, y]:
                    row_array[x, y] = False
                    column_array[x, y] = self.placable_c(x, y)
                else:
                    row_array[x, y] = column_array[x, y] = False
        return row_array, column_array

    def get_graph(self, DTYPE_t[:, :, :] cross_arr, int isgoaly):
        cdef int stack_p, i, dx, dy, xnext, ynext, x, y, x2, y2
        graph = {}
        cdef DTYPE_t[:, :] seen = np.zeros((BOARD_LEN, BOARD_LEN), dtype=DTYPE)
        cdef DTYPE_t[:] stack = np.zeros((BOARD_LEN * BOARD_LEN), dtype=DTYPE)
        for y in range(BOARD_LEN):
            for x in range(BOARD_LEN):
                index = y * BOARD_LEN + x
                if not seen[x, y]:
                    graph[index] = [[], set([]), False, False]
                    stack[0] = index
                    stack_p = 0
                    while stack_p >= 0:
                        index2 = stack[stack_p]
                        stack_p -= 1
                        graph[index][1].add(index2)
                        x2 = index2 % BOARD_LEN
                        y2 = index2 // BOARD_LEN
                        if isgoaly and y2 == 0 or not isgoaly and x2 == 0:
                            graph[index][2] = True
                        if isgoaly and y2 == BOARD_LEN - 1 or not isgoaly and x2 == BOARD_LEN - 1:
                            graph[index][3] = True
                        seen[x2, y2] = 1
                        for i, dx, dy in ((0, 0, -1), (1, 1, 0), (2, 0, 1), (3, -1, 0)):
                            xnext = x2 + dx
                            ynext = y2 + dy
                            if cross_arr[x2, y2, i] and not seen[xnext, ynext]:
                                stack_p += 1
                                stack[stack_p] = ynext * BOARD_LEN + xnext
        return graph

    def placable_cand(self, equivalence_class, graph, int Bx, int By, int Wx, int Wy):
        cdef int x, y
        edges1 = self.get_blocking_edges(equivalence_class[Bx, By], graph, True)
        edges2 = self.get_blocking_edges(equivalence_class[Wx, Wy], graph, False)
        edgess = [edges1, edges2]

        # 実際置けない場所を求める
        cdef np.ndarray[DTYPE_t, ndim = 2] placable = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        for edges in edgess:
            for edge in edges:
                x, y = edge[0]
                if edge[1] == 1:
                    if x > 0:
                        placable[x - 1, y] = 0
                    if x < BOARD_LEN - 1:
                        placable[x, y] = 0
                if edge[1] == 2:
                    placable[x, y] = 0
        return placable

    def calc_placable_row_array(self, np.ndarray[DTYPE_t, ndim = 2] row_cut1, np.ndarray[DTYPE_t, ndim = 2] row_cut2, DTYPE_t[:, :, :] cross_arr, int Bx, int By, int Wx, int Wy, int isgoaly):
        cdef int x, y, index, index2, id1, id2
        global total_time1, total_time2

        #cdef np.ndarray[DTYPE_t, ndim = 2] intersection = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        #intersection[:-1, :] = row_cut2[:-1, :] & row_cut2[1:, :]
        # exception1 = row_cut1[:-1, :] & row_cut1[1:, :]
        # exception2 = 
        #exception = ((row_cut1[:-1, :] & row_cut1[1:, :]) | (~row_cut1[:-1, :] & ~row_cut1[1:, :])) & row_cut2

        # cdef np.ndarray[DTYPE_t, ndim = 2] exception = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        # exception[:-1, :] = row_cut2[:-1, :] & row_cut2[1:, :]
        # exception[1:, :] = exception[1:, :] | (row_cut2[:-1, :] & row_cut2[1:, :])

        # print("a"*30)
        # print(row_cut1)
        # print(row_cut2)
        # print(exception)

        # special_row_cut2 = row_cut2 & exception
        # row_cut2 = row_cut2 & ~exception

        cdef np.ndarray[DTYPE_t, ndim = 2] intersection = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        intersection[:-1, :] = row_cut2[:-1, :] & row_cut2[1:, :]
        exception = (row_cut1[:-1, :] | row_cut1[1:, :]) | intersection
        special_row_cut2 = row_cut2 & exception
        row_cut2 = row_cut2 & ~exception

        for y in range(BOARD_LEN - 1):
            for x in range(BOARD_LEN):
                if row_cut1[x, y]:
                    cross_arr[x, y, DOWN] = 0
                    cross_arr[x, y + 1, UP] = 0
                if x < BOARD_LEN - 1 and row_cut2[x, y]:
                    cross_arr[x, y, DOWN] = 0
                    cross_arr[x, y + 1, UP] = 0
                    cross_arr[x + 1, y, DOWN] = 0
                    cross_arr[x + 1, y + 1, UP] = 0

        start_time = time.time()
        # graph作成
        graph = self.get_graph(cross_arr, isgoaly)  # [edgeリスト, 所属マスset, goal情報1, goal2] ただしy=0含むならgoal1がTrue, y=8ならgoal2がTrue
        total_time2 += time.time() - start_time

        # equivalence class作成
        cdef np.ndarray[DTYPE_t, ndim = 2] equivalence_class = np.zeros((BOARD_LEN, BOARD_LEN), dtype=DTYPE)
        for id1, v in graph.items():
            for id2 in v[1]:
                x = id2 % BOARD_LEN
                y = id2 // BOARD_LEN
                equivalence_class[x, y] = id1
        #print(equivalence_class)

        # edge接続
        for y in range(BOARD_LEN - 1):
            for x in range(BOARD_LEN):
                if row_cut1[x, y]:
                    id1 = equivalence_class[x, y]
                    id2 = equivalence_class[x, y + 1]
                    graph[id1][0].append(Edge(id2, (x, y), 1))
                    graph[id2][0].append(Edge(id1, (x, y), 1))
                elif x < BOARD_LEN - 1 and row_cut2[x, y]:
                    id1s = {equivalence_class[x, y], equivalence_class[x + 1, y]}
                    id2s = {equivalence_class[x, y + 1], equivalence_class[x + 1, y + 1]}
                    if len(id1s) == 1 or len(id2s) == 1:
                        for id1 in id1s:
                            for id2 in id2s:
                                graph[id1][0].append(Edge(id2, (x, y), 2))
                                graph[id2][0].append(Edge(id1, (x, y), 2))
                    else:
                        for id1, id2 in ((equivalence_class[x, y], equivalence_class[x, y + 1]), (equivalence_class[x + 1, y], equivalence_class[x + 1, y + 1])):
                            graph[id1][0].append(Edge(id2, (x, y), 2))
                            graph[id2][0].append(Edge(id1, (x, y), 2))

        placable = self.placable_cand(equivalence_class, graph, Bx, By, Wx, Wy)

        return placable, special_row_cut2, equivalence_class, graph

    def get_all_cycles(self, DTYPE_t[:, :, :] cross_arr):
        cdef int x, y, i
        cdef np.ndarray[DTYPE_t, ndim = 3] direction_array
        cdef np.ndarray[DTYPE_t, ndim = 3] seen = np.zeros((BOARD_LEN, BOARD_LEN, 4), dtype=DTYPE)
        ret = []
        for x in range(BOARD_LEN):
            for y in range(BOARD_LEN):
                for i in range(4):
                    if not seen[x, y, i] and cross_arr[x, y, i] and not cross_arr[x, y, (i - 1) % 4]:
                        direction_array = self.one_cycle(x, y, i, cross_arr)
                        ret.append(direction_array)
                        seen = seen | direction_array
        return ret

    def one_cycle(self, int x, int y, int first_direction, DTYPE_t[:, :, :] cross_arr):
        cdef int x1, y1, x2, y2, isstart, x_start, y_start, direction, index, first_move_direction
        directions = [(UP, 0, -1), (RIGHT, 1, 0), (DOWN, 0, 1), (LEFT, -1, 0)]
        cdef np.ndarray[DTYPE_t, ndim = 3]direction_array = np.zeros((BOARD_LEN, BOARD_LEN, 4), dtype=DTYPE)

        x1 = x
        y1 = y
        x_start = x
        y_start = y
        direction = first_direction
        isstart = True
        while True:
            index = directions[direction][0]
            while not cross_arr[x1, y1, index]:
                direction = (direction + 1) % 4
                index = directions[direction][0]
            if isstart:
                first_move_direction = direction
            direction_array[x1, y1, index] = 1

            if not isstart and x1 == x_start and y1 == y_start and direction == first_move_direction:
                break

            x2 = x1 + directions[direction][1]
            y2 = y1 + directions[direction][2]

            x1 = x2
            y1 = y2

            direction = (direction - 1) % 4
            isstart = False

        return direction_array

    def arrivable_(self, int x, int y, int goal_y, int isleft):
        #cdef np.ndarray[DTYPE_t, ndim = 3] cross_arr
        cdef DTYPE_t[:, :, :] prev = np.zeros((BOARD_LEN, BOARD_LEN, 2), dtype="int32")
        cross = bitarray(4)
        #cross_arr = self.cross_movable_array2(self.row_wall, self.column_wall)  # 一時的にself.row_wallとかを書き換えてこの関数が呼ばれることがあるために毎回計算する必要あり
        x_stack = [x]
        y_stack = [y]
        while len(x_stack) > 0:
            x = x_stack.pop()
            y = y_stack.pop()
            self.seen[x, y] = 1
            if y == goal_y:
                #self.seen = seen
                self.prev = prev
                return True
            #cross = self.cross_movable(x, y)
            cross[0] = (not (y == 0 or self.row_wall_bit[min(x, BOARD_LEN - 2) * 8 + y - 1] or self.row_wall_bit[max(x - 1, 0) * 8 + y - 1]))
            cross[1] = (not (x == BOARD_LEN - 1 or self.column_wall_bit[x * 8 + min(y, BOARD_LEN - 2)] or self.column_wall_bit[x * 8 + max(y - 1, 0)]))
            cross[2] = (not (y == BOARD_LEN - 1 or self.row_wall_bit[min(x, BOARD_LEN - 2) * 8 + y] or self.row_wall_bit[max(x - 1, 0) * 8 + y]))
            cross[3] = (not (x == 0 or self.column_wall_bit[(x - 1) * 8 + min(y, BOARD_LEN - 2)] or self.column_wall_bit[(x - 1) * 8 + max(y - 1, 0)]))
            if isleft and goal_y == 0:
                p_list = [(3, -1, 0), (0, 0, -1), (1, 1, 0), (2, 0, 1)]
            elif not isleft and goal_y == 0:
                p_list = [(2, 0, 1), (1, 1, 0), (0, 0, -1), (3, -1, 0)]
            elif isleft and goal_y == BOARD_LEN - 1:
                p_list = [(1, 1, 0), (2, 0, 1), (3, -1, 0), (0, 0, -1)]
            else:
                p_list = [(0, 0, -1), (3, -1, 0), (2, 0, 1), (1, 1, 0)]
            for i, dx, dy in p_list:
                x2 = x + dx
                y2 = y + dy
                if cross[i] and not self.seen[x2, y2]:
                    prev[x2, y2, 0] = x
                    prev[x2, y2, 1] = y
                    x_stack.append(x2)
                    y_stack.append(y2)
        #self.seen = seen
        self.prev = prev
        return False

    def old_display_cui(self, check_algo=True, official=True, p1_atmark=False):
        sys.stdout.write(" ")
        for c in ["a", "b", "c", "d", "e", "f", "g", "h", "i"]:
            sys.stdout.write(" " + c)
        print("")
        sys.stdout.write(" +")
        for x in range(BOARD_LEN):
            sys.stdout.write("-+")
        print("")

        for y in range(BOARD_LEN):
            if official:
                sys.stdout.write(str(BOARD_LEN - y) + "|")
            else:
                sys.stdout.write(str(y + 1) + "|")
            for x in range(BOARD_LEN):
                if x == self.Bx and y == self.By:
                    if p1_atmark:
                        sys.stdout.write("@")
                    else:
                        sys.stdout.write("O")
                elif x == self.Wx and y == self.Wy:
                    if p1_atmark:
                        sys.stdout.write("O")
                    else:
                        sys.stdout.write("@")
                else:
                    sys.stdout.write(" ")

                if x == BOARD_LEN - 1:
                    break

                if self.column_wall[min(x, BOARD_LEN - 2), min(y, BOARD_LEN - 2)] or self.column_wall[min(x, BOARD_LEN - 2), max(y - 1, 0)]:
                    sys.stdout.write("#")
                else:
                    sys.stdout.write("|")
            print("|")

            if y == BOARD_LEN - 1:
                break

            sys.stdout.write(" +")
            for x in range(BOARD_LEN):
                if self.row_wall[min(x, BOARD_LEN - 2), min(y, BOARD_LEN - 2)] or self.row_wall[max(x - 1, 0), min(y, BOARD_LEN - 2)]:
                    sys.stdout.write("=")
                else:
                    sys.stdout.write("-")

                if x == BOARD_LEN - 1:
                    break

                if self.row_wall[x, y] or self.column_wall[x, y]:
                    sys.stdout.write("+")
                else:
                    sys.stdout.write(" ")
            print("+")

        sys.stdout.write(" +")
        for x in range(BOARD_LEN):
            sys.stdout.write("-+")
        print("")

        print("1p walls:" + str(self.black_walls))
        print("2p walls:" + str(self.white_walls))

        if self.turn % 2 == 0:
            print("{}:1p turn".format(self.turn))
        else:
            print("{}:2p turn".format(self.turn))
        print(total_time1, total_time2)
        # for i in range(4):
        #     print("-"*30)
        #     print(i)
        #     for y in range(BOARD_LEN):
        #         for x in range(BOARD_LEN):
        #             print(self.cross_movable_arr[x, y, i], end=" ")
        #         print("")
        if check_algo:
            self.check_placable_array_algo()

    def display_cui(self, check_algo=True, official=True, p1_atmark=False, ret_str=False):
        ret = " "
        for c in ["a", "b", "c", "d", "e", "f", "g", "h", "i"]:
            ret += " " + c
        ret += os.linesep
        ret += " +"
        for x in range(BOARD_LEN):
            ret += "-+"
        ret += os.linesep

        for y in range(BOARD_LEN):
            if official:
                ret += (str(BOARD_LEN - y) + "|")
            else:
                ret += (str(y + 1) + "|")
            for x in range(BOARD_LEN):
                if x == self.Bx and y == self.By:
                    if p1_atmark:
                        ret += "@"
                    else:
                        ret += "O"
                elif x == self.Wx and y == self.Wy:
                    if p1_atmark:
                        ret += "O"
                    else:
                        ret += "@"
                else:
                    ret += " "

                if x == BOARD_LEN - 1:
                    break

                if self.column_wall[min(x, BOARD_LEN - 2), min(y, BOARD_LEN - 2)] or self.column_wall[min(x, BOARD_LEN - 2), max(y - 1, 0)]:
                    ret += "#"
                else:
                    ret += "|"
            ret += "|" + os.linesep

            if y == BOARD_LEN - 1:
                break

            ret += " +"
            for x in range(BOARD_LEN):
                if self.row_wall[min(x, BOARD_LEN - 2), min(y, BOARD_LEN - 2)] or self.row_wall[max(x - 1, 0), min(y, BOARD_LEN - 2)]:
                    ret += "="
                else:
                    ret += "-"

                if x == BOARD_LEN - 1:
                    break

                if self.row_wall[x, y] or self.column_wall[x, y]:
                    ret += "+"
                else:
                    ret += " "
            ret += "+" + os.linesep

        ret += " +"
        for x in range(BOARD_LEN):
            ret += "-+"
        ret += os.linesep

        ret += "1p walls:" + str(self.black_walls) + os.linesep
        ret += "2p walls:" + str(self.white_walls) + os.linesep

        if self.turn % 2 == 0:
            ret += "{}:1p turn".format(self.turn) + os.linesep
        else:
            ret += "{}:2p turn".format(self.turn) + os.linesep
        # print(total_time1, total_time2)
        # for i in range(4):
        #     print("-"*30)
        #     print(i)
        #     for y in range(BOARD_LEN):
        #         for x in range(BOARD_LEN):
        #             print(self.cross_movable_arr[x, y, i], end=" ")
        #         print("")
        if ret_str:
            return ret
        else:
            sys.stdout.write(ret)

        if check_algo:
            self.check_placable_array_algo()

    def check_placable_array_algo(self):
        placabler1, placablec1 = self.old_calc_placable_array()
        placabler2, placablec2 = self.calc_placable_array()
        #print(np.all(placabler1 == placabler2) and np.all(placablec1 == placablec2))
        if not (np.all(placabler1 == placabler2) and np.all(placablec1 == placablec2)):
            print("")
            print("="*50)
            print("{},{},{},{}".format(self.Bx, self.By, self.Wx, self.Wy))
            for y in range(8):
                for x in range(8):
                    print(int(self.row_wall[x, y]), end="")
                    if x != 7:
                        print(",", end="")
                print("")
            print("")
            for y in range(8):
                for x in range(8):
                    print(int(self.column_wall[x, y]), end="")
                    if x != 7:
                        print(",", end="")
                print("")

            placabler1 = np.array(placabler1, dtype=int)
            placablec1 = np.array(placablec1, dtype=int)
            placabler2 = np.array(placabler2, dtype=int)
            placablec2 = np.array(placablec2, dtype=int)

            print()
            print("row correctness")
            print(placabler1.T == placabler2.T)
            print("row answer")
            print(placabler1.T)
            print("row pred")
            print(placabler2.T)

            print()
            print("column correctness")
            print(placablec1.T == placablec2.T)
            print("column answer")
            print(placablec1.T)
            print("column pred")
            print(placablec2.T)
            print("check_placable_array_algo failed")
            exit()

    def feature_int(self):
        feature = np.zeros((135,), dtype=int)
        feature[0] = self.Bx
        feature[1] = self.By
        feature[2] = self.Wx
        feature[3] = self.Wy
        feature[4] = self.black_walls
        feature[5] = self.white_walls
        feature[6] = self.turn % 2
        feature[7:7 + 64] = self.row_wall.flatten()
        feature[7 + 64:] = self.column_wall.flatten()
        return feature

    def feature(self):
        feature = np.zeros((135,))
        feature[0] = self.Bx / 8
        feature[1] = self.By / 8
        feature[2] = self.Wx / 8
        feature[3] = self.Wy / 8
        feature[4] = self.black_walls / 10
        feature[5] = self.white_walls / 10
        feature[6] = self.turn % 2
        feature[7:7 + 64] = self.row_wall.flatten()
        feature[7 + 64:] = self.column_wall.flatten()
        return feature

    def feature_rev(self):
        feature = np.zeros((135,))
        feature[0] = (7 - self.Bx) / 8
        feature[1] = self.By / 8
        feature[2] = (7 - self.Wx) / 8
        feature[3] = self.Wy / 8
        feature[4] = self.black_walls / 10
        feature[5] = self.white_walls / 10
        feature[6] = self.turn % 2
        feature[7:7 + 64] = np.flip(self.row_wall, 0).flatten()
        feature[7 + 64:] = np.flip(self.column_wall, 0).flatten()
        return feature

    def feature_CNN(self, xflip=False, yflip=False):
        feature = np.zeros((9, 9, CHANNEL))
        Bx = self.Bx
        By = self.By
        Wx = self.Wx
        Wy = self.Wy
        black_walls = self.black_walls
        white_walls = self.white_walls
        turn = self.turn
        row_wall = self.row_wall
        column_wall = self.column_wall
        dist1, dist2 = self.get_player_dist_from_goal()
        cross_arr = np.copy(self.cross_movable_arr)
        placable_r = self.placable_r_
        placable_c = self.placable_c_
        if xflip:
            Bx = 8 - Bx
            Wx = 8 - Wx
            row_wall = np.flip(row_wall, 0)
            column_wall = np.flip(column_wall, 0)
            placable_r = np.flip(placable_r, 0)
            placable_c = np.flip(placable_c, 0)
            cross_arr = np.flip(cross_arr, 0)
            temp = np.copy(cross_arr[:, :, RIGHT])
            cross_arr[:, :, RIGHT] = cross_arr[:, :, LEFT]
            cross_arr[:, :, LEFT] = temp
        if yflip:
            # goalが異なるから色を取り替える
            temp = Bx
            Bx = Wx
            Wx = temp
            By = 8 - self.Wy
            Wy = 8 - self.By
            black_walls = self.white_walls
            white_walls = self.black_walls
            turn += 1
            row_wall = np.flip(row_wall, 1)
            column_wall = np.flip(column_wall, 1)
            placable_r = np.flip(placable_r, 1)
            placable_c = np.flip(placable_c, 1)
            cross_arr = np.flip(cross_arr, 1)
            temp = np.copy(cross_arr[:, :, UP])
            cross_arr[:, :, UP] = cross_arr[:, :, DOWN]
            cross_arr[:, :, DOWN] = temp
            temp = dist1
            dist1 = dist2
            dist2 = temp

        feature[Bx, By, 0] = 1.
        feature[Wx, Wy, 1] = 1.

        feature[:, :, 2] = black_walls / 10
        feature[:, :, 3] = white_walls / 10
        feature[:, :, 4] = turn % 2
        feature[:, :, 5:9] = cross_arr

        feature[:, :, 9] = dist1 / 20
        feature[:, :, 10] = dist2 / 20

        # 以下8*8の特徴量
        feature[:-1, :-1, 11] = row_wall
        feature[:-1, :-1, 12] = column_wall

        feature[:-1, :-1, 13] = placable_r  # 盤面外にはおけないことを考えると、反転しないほうが0paddingと相性は良いと思われる
        feature[:-1, :-1, 14] = placable_c

        return feature

    def feature_CNN_old(self, xflip=False, yflip=False):
        feature = np.zeros((8, 8, 7))
        Bx = self.Bx
        By = self.By
        Wx = self.Wx
        Wy = self.Wy
        black_walls = self.black_walls
        white_walls = self.white_walls
        turn = self.turn
        row_wall = self.row_wall
        column_wall = self.column_wall
        if xflip:
            Bx = 8 - Bx
            Wx = 8 - Wx
            row_wall = np.flip(row_wall, 0)
            column_wall = np.flip(column_wall, 0)
        if yflip:
            # goalが異なるから色を取り替える
            temp = Bx
            Bx = Wx
            Wx = temp
            By = 8 - self.Wy
            Wy = 8 - self.By
            black_walls = self.white_walls
            white_walls = self.black_walls
            turn += 1
            row_wall = np.flip(row_wall, 1)
            column_wall = np.flip(column_wall, 1)

        for x, y in [(Bx, By), (Bx - 1, By), (Bx, By - 1), (Bx - 1, By - 1)]:
            if x < 0 or x >= 8 or y < 0 or y >= 8:
                continue
            feature[x, y, 0] = 1.
        for x, y in [(Wx, Wy), (Wx - 1, Wy), (Wx, Wy - 1), (Wx - 1, Wy - 1)]:
            if x < 0 or x >= 8 or y < 0 or y >= 8:
                continue
            feature[x, y, 1] = 1.

        feature[:, :, 2] = black_walls / 10
        feature[:, :, 3] = white_walls / 10
        feature[:, :, 4] = turn % 2
        feature[:, :, 5] = row_wall
        feature[:, :, 6] = column_wall
        return feature

