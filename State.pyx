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
                ("dist_array1", ctypes.c_uint8 * 81), ("dist_array2", ctypes.c_uint8 * 81),
                ("placable_r_bitarr", ctypes.c_uint64 * 2),
                ("placable_c_bitarr", ctypes.c_uint64 * 2),
                ("terminate", ctypes.c_bool), ("wall0_terminate", ctypes.c_bool), ("pseudo_terminate", ctypes.c_bool),
                ("reward", ctypes.c_int), ("pseudo_reward", ctypes.c_int)]
    
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

calc_dist_array_c = lib.calc_dist_array
calc_dist_array_c.argtypes = [ctypes.POINTER(State_c), ctypes.c_int]
calc_dist_array_c.restype = None

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

movable_array_c = lib.movable_array
movable_array_c.argtypes = [ctypes.POINTER(State_c), ctypes.POINTER(ctypes.c_bool), ctypes.c_int, ctypes.c_int, ctypes.c_bool]
movable_array_c.restype = None

accept_action_str_c = lib.accept_action_str
accept_action_str_c.argtypes = [ctypes.POINTER(State_c), ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
accept_action_str_c.restype = ctypes.c_bool

is_mirror_match = lib.is_mirror_match
is_mirror_match.argtypes = [ctypes.POINTER(State_c)]
is_mirror_match.restype = ctypes.c_bool

calc_placable_array_and_set = lib.calc_placable_array_and_set
calc_placable_array_and_set.argtypes = [ctypes.POINTER(State_c)]
calc_placable_array_and_set.restype = None

uint128ToBoolArray = lib.uint128ToBoolArray
uint128ToBoolArray.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
uint128ToBoolArray.restype = ctypes.POINTER(ctypes.c_bool)

get_player1_dist_from_goal = lib.get_player1_dist_from_goal
get_player1_dist_from_goal.argtypes = [ctypes.POINTER(State_c)]
get_player1_dist_from_goal.restype = ctypes.c_int

get_player2_dist_from_goal = lib.get_player2_dist_from_goal
get_player2_dist_from_goal.argtypes = [ctypes.POINTER(State_c)]
get_player2_dist_from_goal.restype = ctypes.c_int

is_certain_path_terminate_c = lib.is_certain_path_terminate
is_certain_path_terminate_c.argtypes = [ctypes.POINTER(State_c), ctypes.c_int]
is_certain_path_terminate_c.restype = ctypes.c_bool

placable_array_c = lib.placable_array
placable_array_c.argtypes = [ctypes.POINTER(State_c), ctypes.c_int]
placable_array_c.restype = BitArrayPair

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

def movable_array(state, x, y, shortest_only=False):
    mv_c = (ctypes.c_bool * 9)(*([False] * 9))
    movable_array_c(state.state_c, mv_c, x, y, shortest_only)
    mv = np.zeros((3, 3), dtype="bool")
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            mv[dx, dy] = mv_c[(dx + 1) + (dy + 1) * 3]

    return mv

def set_state_by_wall(state):
    # row_wallなどを直接指定して状態を作るときに用いる
    # 差分計算している部分を計算する

    for x in range(BOARD_LEN - 1):
        for y in range(BOARD_LEN - 1):
            if state.row_wall[x, y]:
                set_row_wall_1(state.state_c, x, y)
            else:
                set_row_wall_0(state.state_c, x, y)
            if state.column_wall[x, y]:
                set_column_wall_1(state.state_c, x, y)
            else:
                set_column_wall_0(state.state_c, x, y)

    dist_array1 = calc_dist_array(state, 0)
    dist_array2 = calc_dist_array(state, BOARD_LEN - 1)
    for x in range(BOARD_LEN):
        for y in range(BOARD_LEN):
            state.state_c.dist_array1[x + y * BOARD_LEN] = dist_array1[x, y]
            state.state_c.dist_array2[x + y * BOARD_LEN] = dist_array2[x, y]

    calc_placable_array_and_set(state.state_c)

def get_dist_array_from_c_arr(c_dist_arr):
    return np.array([c_dist_arr[i] for i in range(BOARD_LEN * BOARD_LEN)], dtype=DTYPE).reshape(BOARD_LEN, BOARD_LEN).T

def accept_action_str(state, s, check_placable=True, calc_placable_array=True, check_movable=True):
    # calc_placable_array=Falseにした場合は、以降正しく壁のおける場所を求められないことに注意

    old_wall_num = state.black_walls + state.white_walls
    
    ret = accept_action_str_c(state.state_c, s.encode('utf-8'), check_placable, calc_placable_array, check_movable)

    state.Bx = state.state_c.Bx
    state.By = state.state_c.By
    state.Wx = state.state_c.Wx
    state.Wy = state.state_c.Wy
    state.turn = state.state_c.turn
    state.black_walls = state.state_c.black_walls
    state.white_walls = state.state_c.white_walls

    state.terminate = state.state_c.terminate
    state.reward = state.state_c.reward
    state.wall0_terminate = state.state_c.wall0_terminate
    state.pseudo_terminate = state.state_c.pseudo_terminate
    state.pseudo_reward = state.state_c.pseudo_reward

    # 壁置きなら
    if old_wall_num != state.black_walls + state.white_walls:
        state.row_wall = get_numpy_arr(state.state_c.row_wall_bitarr, BOARD_LEN - 1)
        state.column_wall = get_numpy_arr(state.state_c.column_wall_bitarr, BOARD_LEN - 1)

    return ret

def get_player_dist_from_goal(state):
    return get_player1_dist_from_goal(state.state_c), get_player2_dist_from_goal(state.state_c)

def is_certain_path_terminate(state, color=-1):
    return is_certain_path_terminate_c(state.state_c, color)

def placable_array(state, color):
    ret = placable_array_c(state.state_c, color)
    return get_numpy_arr(ret.bitarr1, BOARD_LEN - 1), get_numpy_arr(ret.bitarr2, BOARD_LEN - 1)

def calc_dist_array(state, goal_y):
    calc_dist_array_c(state.state_c, goal_y)
    if goal_y == 0:
        array_ptr = state.state_c.dist_array1
    else:
        array_ptr = state.state_c.dist_array2
    return np.array([array_ptr[i] for i in range(BOARD_LEN * BOARD_LEN)], dtype=DTYPE).reshape(BOARD_LEN, BOARD_LEN).T

# -----------------------------------------

def get_numpy_arr(bitarr, int len_, int offset=0):
    cdef np.ndarray[DTYPE_t, ndim = 2] ret
    cdef int x, y
    ret = np.zeros((len_, len_), dtype=DTYPE)
    bool_p = uint128ToBoolArray(bitarr[1 + offset], bitarr[0 + offset])
    for x in range(len_):
        for y in range(len_):
            ret[x, y] = bool_p[x + y * BIT_BOARD_LEN]
    return ret


def print_bitarr(bitarr):
    print()
    for y in range(BOARD_LEN):
        for x in range(BOARD_LEN):
            print(bitarr[x + y * BIT_BOARD_LEN], end="")
        print()

cdef class State:
    draw_turn = DRAW_TURN
    cdef public np.ndarray row_wall, column_wall
    cdef public int Bx, By, Wx, Wy, turn, black_walls, white_walls, terminate, reward, wall0_terminate, pseudo_terminate, pseudo_reward
    cdef public state_c
    def __init__(self):
        self.row_wall = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.column_wall = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
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

        self.state_c = State_c()
        State_init_(self.state_c)
        # print_state(self.state_c)

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

        cross_arr = np.zeros((9, 9, 4))
        for i in range(4):
            cross_arr[:, :, i] = get_numpy_arr(self.state_c.cross_bitarrs, BOARD_LEN, i * 2)
            
        Bx = self.Bx
        By = self.By
        Wx = self.Wx
        Wy = self.Wy
        black_walls = self.black_walls
        white_walls = self.white_walls
        turn = self.turn
        row_wall = self.row_wall
        column_wall = self.column_wall
        dist1, dist2 = get_player_dist_from_goal(self)
        placable_r = get_numpy_arr(self.state_c.placable_r_bitarr, BOARD_LEN - 1)
        placable_c = get_numpy_arr(self.state_c.placable_c_bitarr, BOARD_LEN - 1)
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

