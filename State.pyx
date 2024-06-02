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
    
class State(ctypes.Structure):
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

State_init = lib.State_init
State_init.argtypes = [ctypes.POINTER(State)]
State_init.restype = None

# dll中の関数の引数と戻り値の型を指定
arrivable_ = lib.arrivable_
arrivable_.argtypes = [ctypes.POINTER(State), ctypes.c_int, ctypes.c_int, ctypes.c_int]
arrivable_.restype = ctypes.c_bool

calc_placable_array_ = lib.calc_placable_array_
calc_placable_array_.argtypes = [ctypes.POINTER(State)]
calc_placable_array_.restype = BitArrayPair

calc_dist_array_c = lib.calc_dist_array
calc_dist_array_c.argtypes = [ctypes.POINTER(State), ctypes.c_int]
calc_dist_array_c.restype = None

print_state = lib.print_state
print_state.argtypes = [ctypes.POINTER(State)]
print_state.restype = None

set_row_wall_1 = lib.set_row_wall_1
set_row_wall_1.argtypes = [ctypes.POINTER(State), ctypes.c_int, ctypes.c_int]
set_row_wall_1.restype = None
set_row_wall_0 = lib.set_row_wall_0
set_row_wall_0.argtypes = [ctypes.POINTER(State), ctypes.c_int, ctypes.c_int]
set_row_wall_0.restype = None

set_column_wall_1 = lib.set_column_wall_1
set_column_wall_1.argtypes = [ctypes.POINTER(State), ctypes.c_int, ctypes.c_int]
set_column_wall_1.restype = None
set_column_wall_0 = lib.set_column_wall_0
set_column_wall_0.argtypes = [ctypes.POINTER(State), ctypes.c_int, ctypes.c_int]
set_column_wall_0.restype = None

eq_state = lib.eq_state
eq_state.argtypes = [ctypes.POINTER(State), ctypes.POINTER(State)]
eq_state.restype = ctypes.c_bool

color_p_c = lib.color_p
color_p_c.argtypes = [ctypes.POINTER(State), ctypes.c_int]
color_p_c.restype = Point_c

movable_array_c = lib.movable_array
movable_array_c.argtypes = [ctypes.POINTER(State), ctypes.POINTER(ctypes.c_bool), ctypes.c_int, ctypes.c_int, ctypes.c_bool]
movable_array_c.restype = None

accept_action_str_c = lib.accept_action_str
accept_action_str_c.argtypes = [ctypes.POINTER(State), ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
accept_action_str_c.restype = ctypes.c_bool

is_mirror_match = lib.is_mirror_match
is_mirror_match.argtypes = [ctypes.POINTER(State)]
is_mirror_match.restype = ctypes.c_bool

calc_placable_array_and_set = lib.calc_placable_array_and_set
calc_placable_array_and_set.argtypes = [ctypes.POINTER(State)]
calc_placable_array_and_set.restype = None

uint128ToBoolArray = lib.uint128ToBoolArray
uint128ToBoolArray.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
uint128ToBoolArray.restype = ctypes.POINTER(ctypes.c_bool)

get_player1_dist_from_goal = lib.get_player1_dist_from_goal
get_player1_dist_from_goal.argtypes = [ctypes.POINTER(State)]
get_player1_dist_from_goal.restype = ctypes.c_int

get_player2_dist_from_goal = lib.get_player2_dist_from_goal
get_player2_dist_from_goal.argtypes = [ctypes.POINTER(State)]
get_player2_dist_from_goal.restype = ctypes.c_int

is_certain_path_terminate_c = lib.is_certain_path_terminate
is_certain_path_terminate_c.argtypes = [ctypes.POINTER(State), ctypes.c_int]
is_certain_path_terminate_c.restype = ctypes.c_bool

placable_array_c = lib.placable_array
placable_array_c.argtypes = [ctypes.POINTER(State), ctypes.c_int]
placable_array_c.restype = BitArrayPair

# -------------------
# arrayで返す必要があるなどの理由でラップしている関数を以下に定義


def color_p(state, color):
    ret = color_p_c(state, color)
    return ret.x, ret.y


def movable_array(state, x, y, shortest_only=False):
    cdef np.ndarray[DTYPE_t, ndim = 2] mv
    mv_c = (ctypes.c_bool * 9)(*([False] * 9))
    movable_array_c(state, mv_c, x, y, shortest_only)
    mv = np.zeros((3, 3), dtype=DTYPE)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            mv[dx, dy] = mv_c[(dx + 1) + (dy + 1) * 3]

    return mv


def set_wall(state, row_wall, column_wall):
    for x in range(BOARD_LEN - 1):
        for y in range(BOARD_LEN - 1):
            if row_wall[x, y]:
                set_row_wall_1(state, x, y)
            else:
                set_row_wall_0(state, x, y)
            if column_wall[x, y]:
                set_column_wall_1(state, x, y)
            else:
                set_column_wall_0(state, x, y)


def set_state_by_wall(state, row_wall, column_wall):
    # 差分計算している部分を計算する

    set_wall(state, row_wall, column_wall)

    dist_array1 = calc_dist_array(state, 0)
    dist_array2 = calc_dist_array(state, BOARD_LEN - 1)
    for x in range(BOARD_LEN):
        for y in range(BOARD_LEN):
            state.dist_array1[x + y * BOARD_LEN] = dist_array1[x, y]
            state.dist_array2[x + y * BOARD_LEN] = dist_array2[x, y]

    calc_placable_array_and_set(state)


def get_dist_array_from_c_arr(c_dist_arr):
    return np.array([c_dist_arr[i] for i in range(BOARD_LEN * BOARD_LEN)], dtype=DTYPE).reshape(BOARD_LEN, BOARD_LEN).T


def accept_action_str(state, s, check_placable=True, calc_placable_array=True, check_movable=True):
    # calc_placable_array=Falseにした場合は、以降正しく壁のおける場所を求められないことに注意
    return accept_action_str_c(state, s.encode('utf-8'), check_placable, calc_placable_array, check_movable)


def get_player_dist_from_goal(state):
    return get_player1_dist_from_goal(state), get_player2_dist_from_goal(state)


def is_certain_path_terminate(state, color=-1):
    return is_certain_path_terminate_c(state, color)


def placable_array(state, color):
    ret = placable_array_c(state, color)
    return get_numpy_arr(ret.bitarr1, BOARD_LEN - 1), get_numpy_arr(ret.bitarr2, BOARD_LEN - 1)


def placable_flatten_array(state, color):
    ret = placable_array_c(state, color)
    return get_flatten_numpy_arr(ret.bitarr1, BOARD_LEN - 1), get_flatten_numpy_arr(ret.bitarr2, BOARD_LEN - 1)
    

def calc_dist_array(state, goal_y):
    calc_dist_array_c(state, goal_y)
    if goal_y == 0:
        array_ptr = state.dist_array1
    else:
        array_ptr = state.dist_array2
    return np.array([array_ptr[i] for i in range(BOARD_LEN * BOARD_LEN)], dtype=DTYPE).reshape(BOARD_LEN, BOARD_LEN).T


def display_cui(state, check_algo=True, official=True, p1_atmark=False, ret_str=False):
    row_wall = get_numpy_arr(state.row_wall_bitarr, BOARD_LEN - 1)
    column_wall = get_numpy_arr(state.column_wall_bitarr, BOARD_LEN - 1)
    
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
            if x == state.Bx and y == state.By:
                if p1_atmark:
                    ret += "@"
                else:
                    ret += "O"
            elif x == state.Wx and y == state.Wy:
                if p1_atmark:
                    ret += "O"
                else:
                    ret += "@"
            else:
                ret += " "

            if x == BOARD_LEN - 1:
                break

            if column_wall[min(x, BOARD_LEN - 2), min(y, BOARD_LEN - 2)] or column_wall[min(x, BOARD_LEN - 2), max(y - 1, 0)]:
                ret += "#"
            else:
                ret += "|"
        ret += "|" + os.linesep

        if y == BOARD_LEN - 1:
            break

        ret += " +"
        for x in range(BOARD_LEN):
            if row_wall[min(x, BOARD_LEN - 2), min(y, BOARD_LEN - 2)] or row_wall[max(x - 1, 0), min(y, BOARD_LEN - 2)]:
                ret += "="
            else:
                ret += "-"

            if x == BOARD_LEN - 1:
                break

            if row_wall[x, y] or column_wall[x, y]:
                ret += "+"
            else:
                ret += " "
        ret += "+" + os.linesep

    ret += " +"
    for x in range(BOARD_LEN):
        ret += "-+"
    ret += os.linesep

    ret += "1p walls:" + str(state.black_walls) + os.linesep
    ret += "2p walls:" + str(state.white_walls) + os.linesep

    if state.turn % 2 == 0:
        ret += "{}:1p turn".format(state.turn) + os.linesep
    else:
        ret += "{}:2p turn".format(state.turn) + os.linesep
    if ret_str:
        return ret
    else:
        sys.stdout.write(ret)


def feature_int(state):
    row_wall = get_numpy_arr(state.row_wall_bitarr, BOARD_LEN - 1)
    column_wall = get_numpy_arr(state.column_wall_bitarr, BOARD_LEN - 1)
    feature = np.zeros((135,), dtype=int)
    feature[0] = state.Bx
    feature[1] = state.By
    feature[2] = state.Wx
    feature[3] = state.Wy
    feature[4] = state.black_walls
    feature[5] = state.white_walls
    feature[6] = state.turn % 2
    feature[7:7 + 64] = row_wall.flatten()
    feature[7 + 64:] = column_wall.flatten()
    return feature


def feature_CNN(state, xflip=False, yflip=False):
    feature = np.zeros((9, 9, CHANNEL))

    cross_arr = np.zeros((9, 9, 4))
    for i in range(4):
        cross_arr[:, :, i] = get_numpy_arr(state.cross_bitarrs, BOARD_LEN, i * 2)
        
    Bx = state.Bx
    By = state.By
    Wx = state.Wx
    Wy = state.Wy
    black_walls = state.black_walls
    white_walls = state.white_walls
    turn = state.turn
    row_wall = get_numpy_arr(state.row_wall_bitarr, BOARD_LEN - 1)
    column_wall = get_numpy_arr(state.column_wall_bitarr, BOARD_LEN - 1)
    dist1, dist2 = get_player_dist_from_goal(state)
    placable_r = get_numpy_arr(state.placable_r_bitarr, BOARD_LEN - 1)
    placable_c = get_numpy_arr(state.placable_c_bitarr, BOARD_LEN - 1)
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
        By = 8 - state.Wy
        Wy = 8 - state.By
        black_walls = state.white_walls
        white_walls = state.black_walls
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


def get_row_wall(state):
    return get_numpy_arr(state.row_wall_bitarr, BOARD_LEN - 1)


def get_column_wall(state):
    return get_numpy_arr(state.column_wall_bitarr, BOARD_LEN - 1)


def get_numpy_arr(bitarr, int len_, int offset=0):
    cdef np.ndarray[DTYPE_t, ndim = 2] ret
    cdef int x, y
    ret = np.zeros((len_, len_), dtype=DTYPE)
    bool_p = uint128ToBoolArray(bitarr[1 + offset], bitarr[0 + offset])
    for x in range(len_):
        for y in range(len_):
            ret[x, y] = bool_p[x + y * BIT_BOARD_LEN]
    return ret


def get_flatten_numpy_arr(bitarr, int len_, int offset=0):
    cdef np.ndarray[DTYPE_t, ndim = 1] ret
    cdef int i, x, y
    ret = np.zeros(len_ * len_, dtype=DTYPE)
    bool_p = uint128ToBoolArray(bitarr[1 + offset], bitarr[0 + offset])
    i = 0
    for x in range(len_):
        for y in range(len_):
            ret[i] = bool_p[x + y * BIT_BOARD_LEN]
            i += 1
    return ret


def print_bitarr(bitarr):
    print()
    for y in range(BOARD_LEN):
        for x in range(BOARD_LEN):
            print(bitarr[x + y * BIT_BOARD_LEN], end="")
        print()

