from sys import exit
# coding: utf-8
# cython: language_level=3, boundscheck=False
# cython: profile=True
import numpy as np
cimport numpy as np
import sys, os
import time
import copy
import collections
import math
from bitarray import bitarray
from bitarray.util import ba2int, int2ba
import ctypes

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
BITARRAY_SIZE = 128
BIT_BOARD_LEN = 11

# ------------------------------
# c++の関数群の定義

if os.name == "nt":
    lib = ctypes.CDLL('./State_util.dll')
else:
    lib = ctypes.CDLL('./State_util.so')

# C++で定義された構造体に対応するPythonクラスを定義
class BitArrayPair(ctypes.Structure):
    _fields_ = [("bitarr1", ctypes.c_uint64 * 2),  # __uint128_tを2つのuint64として扱う。bitarr1[0]に右側の64bit、bitarr1[1]に左側の64bitが格納されることに注意する
                ("bitarr2", ctypes.c_uint64 * 2)]
    
class State_c(ctypes.Structure):
    _fields_ = [("row_wall_bitarr", ctypes.c_uint64 * 2),
                ("column_wall_bitarr", ctypes.c_uint64 * 2),
                ("cross_bitarrs", ctypes.c_uint64 * 8),
                ("Bx", ctypes.c_int), ("By", ctypes.c_int), ("Wx", ctypes.c_int), ("Wy", ctypes.c_int),
                ("turn", ctypes.c_int),
                ("black_walls", ctypes.c_int), ("white_walls", ctypes.c_int),
                ("dist_array1", ctypes.c_uint8 * 81), ("dist_array2", ctypes.c_uint8 * 81)]
    
class Point_c(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int)]

State_init_ = lib.State_init
State_init_.argtypes = [ctypes.POINTER(State_c)]
State_init_.restype = None

# dll中の関数の引数と戻り値の型を指定
arrivable_ = lib.arrivable_
arrivable_.argtypes = [ctypes.POINTER(State_c), ctypes.c_int, ctypes.c_int, ctypes.c_int]
arrivable_.restype = ctypes.c_bool

calc_placable_array_ = lib.calc_placable_array_
calc_placable_array_.argtypes = [ctypes.POINTER(State_c)]
calc_placable_array_.restype = BitArrayPair

calc_dist_array = lib.calc_dist_array
calc_dist_array.argtypes = [ctypes.POINTER(State_c), ctypes.c_int]
calc_dist_array.restype = None

print_state = lib.print_state
print_state.argtypes = [ctypes.POINTER(State_c)]
print_state.restype = None

set_row_wall_1 = lib.set_row_wall_1
set_row_wall_1.argtypes = [ctypes.POINTER(State_c), ctypes.c_int, ctypes.c_int]
set_row_wall_1.restype = None
set_row_wall_0 = lib.set_row_wall_0
set_row_wall_0.argtypes = [ctypes.POINTER(State_c), ctypes.c_int, ctypes.c_int]
set_row_wall_0.restype = None

set_column_wall_1 = lib.set_column_wall_1
set_column_wall_1.argtypes = [ctypes.POINTER(State_c), ctypes.c_int, ctypes.c_int]
set_column_wall_1.restype = None
set_column_wall_0 = lib.set_column_wall_0
set_column_wall_0.argtypes = [ctypes.POINTER(State_c), ctypes.c_int, ctypes.c_int]
set_column_wall_0.restype = None

eq_state_c = lib.eq_state
eq_state_c.argtypes = [ctypes.POINTER(State_c), ctypes.POINTER(State_c)]
eq_state_c.restype = ctypes.c_bool

color_p_c = lib.color_p
color_p_c.argtypes = [ctypes.POINTER(State_c), ctypes.c_int]
color_p_c.restype = Point_c

accept_action_str_c = lib.accept_action_str
accept_action_str_c.argtypes = [ctypes.POINTER(State_c), ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
accept_action_str_c.restype = ctypes.c_bool

is_mirror_match = lib.is_mirror_match
is_mirror_match.argtypes = [ctypes.POINTER(State_c)]
is_mirror_match.restype = ctypes.c_bool

# -------------------------------------------
# TODO: 以下、State_util.cppの実装が完了したらすべてそれに置き換える。一時的な関数。

def State_init(state):
    pass  # pythonではコンストラクタで初期化されるから何もしない

def eq_state(state1, state2):
    f = np.all(state1.row_wall == state2.row_wall) and np.all(state1.column_wall == state2.column_wall)
    f = f and state1.Bx == state2.Bx and state1.By == state2.By and state1.Wx == state2.Wx and state1.Wy == state2.Wy
    f = f and state1.black_walls == state2.black_walls and state1.white_walls == state2.white_walls
    f = f and eq_state_c(state1.state_c, state2.state_c)
    return f

# c++で直接２つのintを返させてそのままpythonでも２つのintを返すのは難しそう。この関数はこのまま使う。
def color_p(state, color):
    ret = color_p_c(state.state_c, color)
    return ret.x, ret.y

# -----------------------------------------

def print_bitarr(bitarr):
    print()
    for y in range(BOARD_LEN):
        for x in range(BOARD_LEN):
            print(bitarr[x + y * BIT_BOARD_LEN], end="")
        print()

cdef class State:
    draw_turn = DRAW_TURN
    cdef public np.ndarray seen, row_wall, column_wall, must_be_checked_x, must_be_checked_y, placable_r_, placable_c_, placable_rb, placable_cb, placable_rw, placable_cw, dist_array1, dist_array2
    cdef public int Bx, By, Wx, Wy, turn, black_walls, white_walls, terminate, reward, wall0_terminate, pseudo_terminate, pseudo_reward
    cdef public DTYPE_t[:, :, :] prev, cross_movable_arr
    cdef public state_c
    def __init__(self):
        self.row_wall = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.column_wall = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
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
        self.dist_array1 = np.zeros((BOARD_LEN, BOARD_LEN), dtype="int8")
        self.dist_array2 = np.zeros((BOARD_LEN, BOARD_LEN), dtype="int8")

        self.state_c = State_c()
        State_init_(self.state_c)
        # print_state(self.state_c)

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
        assert False
        return False

    def get_player_dist_from_goal(self):
        return self.dist_array1[self.Bx, self.By], self.dist_array2[self.Wx, self.Wy]

    def color_p(self, color):
        assert False

    def accept_action_str(self, s, check_placable=True, calc_placable_array=True, check_movable=True):
        # calc_placable_array=Falseにした場合は、以降正しく壁のおける場所を求められないことに注意
        
        accept_action_str_c(self.state_c, s.encode('utf-8'), check_placable, calc_placable_array, check_movable)

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
                self.Bx = self.state_c.Bx = x
                self.By = self.state_c.By = y
                
            else:
                self.Wx = self.state_c.Wx = x
                self.Wy = self.state_c.Wy = y

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
                        set_row_wall_1(self.state_c, x, y)
                        if self.turn % 2 == 0:
                            self.black_walls -= 1
                            self.state_c.black_walls -= 1
                        else:
                            self.white_walls -= 1
                            self.state_c.white_walls -= 1
                    else:
                        return False
                elif s[2] == "v":
                    if cf and walls >= 1:
                        self.column_wall[x, y] = 1
                        set_column_wall_1(self.state_c, x, y)
                        if self.turn % 2 == 0:
                            self.black_walls -= 1
                            self.state_c.black_walls -= 1
                        else:
                            self.white_walls -= 1
                            self.state_c.white_walls -= 1
                    else:
                        return False
                else:
                    return False
            else:  # 非合法手が来ない前提で、placableを省略して高速化
                if s[2] == "h":
                    if walls >= 1:
                        self.row_wall[x, y] = 1
                        set_row_wall_1(self.state_c, x, y)
                        if self.turn % 2 == 0:
                            self.black_walls -= 1
                            self.state_c.black_walls -= 1
                        else:
                            self.white_walls -= 1
                            self.state_c.white_walls -= 1
                    else:
                        return False
                elif s[2] == "v":
                    if walls >= 1:
                        self.column_wall[x, y] = 1
                        set_column_wall_1(self.state_c, x, y)
                        if self.turn % 2 == 0:
                            self.black_walls -= 1
                            self.state_c.black_walls -= 1
                        else:
                            self.white_walls -= 1
                            self.state_c.white_walls -= 1
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
                self.dist_array1 = self.calc_dist_array(0)
                self.dist_array2 = self.calc_dist_array(BOARD_LEN - 1)
        self.turn += 1
        self.state_c.turn += 1

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
            elif is_mirror_match(self.state_c):
                self.pseudo_terminate = True
                self.pseudo_reward = -1
            else:
                self.pseudo_terminate = False
                self.pseudo_reward = 0

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

    def set_state_by_wall(self):
        # row_wallなどを直接指定して状態を作るときに用いる
        # 差分計算している部分を計算する

        for x in range(BOARD_LEN - 1):
            for y in range(BOARD_LEN - 1):
                if self.row_wall[x, y]:
                    set_row_wall_1(self.state_c, x, y)
                else:
                    set_row_wall_0(self.state_c, x, y)
                if self.column_wall[x, y]:
                    set_column_wall_1(self.state_c, x, y)
                else:
                    set_column_wall_0(self.state_c, x, y)

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
                dist_arr = self.calc_dist_array(0 if self.turn % 2 == 0 else BOARD_LEN - 1)
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
            set_row_wall_1(self.state_c, x, y)
            if color == 0:
                f = self.arrivable(self.Bx, self.By, 0) 
            else:
                f = self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.row_wall[x, y] = 0
            set_row_wall_0(self.state_c, x, y)
            row_f = row_f and f
        if column_f:
            self.column_wall[x, y] = 1
            set_column_wall_1(self.state_c, x, y)
            if color == 0:
                f = self.arrivable(self.Bx, self.By, 0)
            else:
                f = self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.column_wall[x, y] = 0
            set_column_wall_0(self.state_c, x, y)
            column_f = column_f and f
        return row_f, column_f
    
    cdef (int, int) placable(self, x, y):
        if self.row_wall[x, y] or self.column_wall[x, y]:
            return False, False
        row_f = True
        column_f = True

        if self.row_wall[max(x - 1, 0), y] or self.row_wall[min(x + 1, BOARD_LEN - 2), y]:
            row_f = False
        if self.column_wall[x, max(y - 1, 0)] or self.column_wall[x, min(y + 1, BOARD_LEN - 2)]:
            column_f = False
        if row_f and self.must_be_checked_y[x, y]:
            self.row_wall[x, y] = 1
            set_row_wall_1(self.state_c, x, y)
            f = self.arrivable(self.Bx, self.By, 0) and self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.row_wall[x, y] = 0
            set_row_wall_0(self.state_c, x, y)
            row_f = row_f and f
        if column_f and self.must_be_checked_x[x, y]:
            self.column_wall[x, y] = 1
            set_column_wall_1(self.state_c, x, y)
            f = self.arrivable(self.Bx, self.By, 0) and self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.column_wall[x, y] = 0
            set_column_wall_0(self.state_c, x, y)
            column_f = column_f and f
        return row_f, column_f

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

    cdef np.ndarray[DTYPE_t, ndim = 2] calc_dist_array(self, int goal_y):
        calc_dist_array(self.state_c, goal_y)
        if goal_y == 0:
            array_ptr = self.state_c.dist_array1
        else:
            array_ptr = self.state_c.dist_array2
        return np.array([array_ptr[i] for i in range(BOARD_LEN * BOARD_LEN)], dtype=DTYPE).reshape(BOARD_LEN, BOARD_LEN).T
    
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
    
    def calc_placable_array(self, skip_calc_graph=False):
        cdef np.ndarray[DTYPE_t, ndim = 2] row_array = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 2] column_array = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef int x, y

        ret = calc_placable_array_(self.state_c)

        row_placable_bitarr = bitarray(128)
        column_placable_bitarr = bitarray(128)
        row_placable_bitarr[:64] = int2ba(ret.bitarr1[1], length=64)
        row_placable_bitarr[64:] = int2ba(ret.bitarr1[0], length=64)
        column_placable_bitarr[:64] = int2ba(ret.bitarr2[1], length=64)
        column_placable_bitarr[64:] = int2ba(ret.bitarr2[0], length=64)
        # print_bitarr(row_placable_bitarr)
        # print_bitarr(column_placable_bitarr)

        for x in range(BOARD_LEN - 1):
            for y in range(BOARD_LEN - 1):
                row_array[x, y] = row_placable_bitarr[x + y * BIT_BOARD_LEN]
                column_array[x, y] = column_placable_bitarr[x + y * BIT_BOARD_LEN]

        return row_array, column_array

    def arrivable(self, int x, int y, int goal_y):
        return arrivable_(self.state_c, x, y, goal_y)

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

