# coding:utf-8
#cython: language_level=3, boundscheck=False
# cython: profile=True
import numpy as np
cimport numpy as np
import sys
import time
import copy
import collections

DTYPE = np.int
ctypedef np.int_t DTYPE_t

BOARD_LEN = 9
DRAW_TURN = 200
CHANNEL = 11
CALC_DIST_ARRAY = True
notation_dict = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8}
total_time1 = 0.
total_time2 = 0.
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


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
    cdef public np.ndarray seen, row_wall, column_wall, must_be_checked_x, must_be_checked_y, prev, placable_rb, placable_cb, placable_rw, placable_cw, row_special_cut, row_eq, column_special_cut, column_eq, dist_array1, dist_array2
    cdef public int Bx, By, Wx, Wy, turn, black_walls, white_walls, terminate, reward
    cdef public row_graph, column_graph
    def __init__(self):
        self.row_wall = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.column_wall = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
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
        self.reward = 0  # blackから見たreward
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

    def accept_action_str(self, s):
        cdef np.ndarray[DTYPE_t, ndim = 3] cross_arr
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
            mv = self.movable_array(x2, y2)
            dx = x - x2
            dy = y - y2
            if abs(dx) + abs(dy) >= 3:
                return False
            jump = False
            if abs(dx) == 2 or abs(dy) == 2:
                jump = True
                x3 = x2 + dx // 2
                y3 = y2 + dy // 2
                if not ((self.Bx == x3 and self.By == y3) or (self.Wx == x3 and self.Wy == y3)):
                    return False
                dx //= 2
                dy //= 2
            if not mv[dx, dy]:
                return False
            if self.turn % 2 == 0:
                x0 = self.Bx
                y0 = self.By
                self.Bx = x
                self.By = y
            else:
                x0 = self.Wx
                y0 = self.Wy
                self.Wx = x
                self.Wy = y

            # placable_arrayを更新
            placable_r, placable_c = self.calc_placable_array(skip_calc_graph=True)
            self.placable_rb, self.placable_cb = (placable_r * (self.black_walls >= 1), placable_c * self.black_walls >= 1)
            self.placable_rw, self.placable_cw = (placable_r * (self.white_walls >= 1), placable_c * self.white_walls >= 1)

        elif len(s) == 3:
            rf, cf = self.placable(x, y)
            if self.turn % 2 == 0:
                walls = self.black_walls
            else:
                walls = self.white_walls
            if s[2] == "h":
                if rf and walls >= 1:
                    self.row_wall[x, y] = 1
                    if self.turn % 2 == 0:
                        self.black_walls -= 1
                    else:
                        self.white_walls -= 1
                else:
                    return False
            elif s[2] == "v":
                if cf and walls >= 1:
                    self.column_wall[x, y] = 1
                    if self.turn % 2 == 0:
                        self.black_walls -= 1
                    else:
                        self.white_walls -= 1
                else:
                    return False
            else:
                return False

            # 壁置きの場合計算しなおし
            placable_r, placable_c = self.calc_placable_array()
            self.placable_rb, self.placable_cb = (placable_r * (self.black_walls >= 1), placable_c * self.black_walls >= 1)
            self.placable_rw, self.placable_cw = (placable_r * (self.white_walls >= 1), placable_c * self.white_walls >= 1)

            # dist_arrayも計算しなおし
            if CALC_DIST_ARRAY:
                cross_arr = self.cross_movable_array2(self.row_wall, self.column_wall)
                self.dist_array1 = self.dist_array(0, cross_arr)
                self.dist_array2 = self.dist_array(BOARD_LEN - 1, cross_arr)
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
        return True

    # 上，右，下，左
    def cross_movable(self, x, y):
        ret = np.zeros((4,), dtype="bool")
        ret[UP] = (not (y == 0 or self.row_wall[min(x, BOARD_LEN - 2), y - 1] or self.row_wall[max(x - 1, 0), y - 1]))
        ret[RIGHT] = (not (x == BOARD_LEN - 1 or self.column_wall[x, min(y, BOARD_LEN - 2)] or self.column_wall[x, max(y - 1, 0)]))
        ret[DOWN] = (not (y == BOARD_LEN - 1 or self.row_wall[min(x, BOARD_LEN - 2), y] or self.row_wall[max(x - 1, 0), y]))
        ret[LEFT] = (not (x == 0 or self.column_wall[x - 1, min(y, BOARD_LEN - 2)] or self.column_wall[x - 1, max(y - 1, 0)]))
        return ret

    def cross_movable_array(self):
        cdef int x, y
        cdef np.ndarray[DTYPE_t, ndim = 3] ret = np.zeros((BOARD_LEN, BOARD_LEN, 4), dtype=DTYPE)
        for x in range(BOARD_LEN):
            for y in range(BOARD_LEN):
                ret[x, y, UP] = (not (y == 0 or self.row_wall[min(x, BOARD_LEN - 2), y - 1] or self.row_wall[max(x - 1, 0), y - 1]))
                ret[x, y, RIGHT] = (not (x == BOARD_LEN - 1 or self.column_wall[x, min(y, BOARD_LEN - 2)] or self.column_wall[x, max(y - 1, 0)]))
                ret[x, y, DOWN] = (not (y == BOARD_LEN - 1 or self.row_wall[min(x, BOARD_LEN - 2), y] or self.row_wall[max(x - 1, 0), y]))
                ret[x, y, LEFT] = (not (x == 0 or self.column_wall[x - 1, min(y, BOARD_LEN - 2)] or self.column_wall[x - 1, max(y - 1, 0)]))
        return ret

    def cross_movable_array2(self, row_wall, column_wall):
        cdef int x, y
        cdef np.ndarray[DTYPE_t, ndim = 3] ret = np.zeros((BOARD_LEN, BOARD_LEN, 4), dtype=DTYPE)
        for x in range(BOARD_LEN):
            for y in range(BOARD_LEN):
                ret[x, y, UP] = (not (y == 0 or row_wall[min(x, BOARD_LEN - 2), y - 1] or row_wall[max(x - 1, 0), y - 1]))
                ret[x, y, RIGHT] = (not (x == BOARD_LEN - 1 or column_wall[x, min(y, BOARD_LEN - 2)] or column_wall[x, max(y - 1, 0)]))
                ret[x, y, DOWN] = (not (y == BOARD_LEN - 1 or row_wall[min(x, BOARD_LEN - 2), y] or row_wall[max(x - 1, 0), y]))
                ret[x, y, LEFT] = (not (x == 0 or column_wall[x - 1, min(y, BOARD_LEN - 2)] or column_wall[x - 1, max(y - 1, 0)]))
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
                dist_arr = self.dist_array(0 if self.turn % 2 == 0 else BOARD_LEN - 1, self.cross_movable_array())
        for i, p in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            if not cross[i]:
                continue
            x2 = x + p[0]
            y2 = y + p[1]
            if (self.Bx == x2 and self.By == y2) or (self.Wx == x2 and self.Wy == y2):
                cross2 = self.cross_movable(x2, y2)
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
            f = self.arrivable(self.Bx, self.By, 0) and self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.row_wall[x, y] = 0
            row_f = row_f and f
        if column_f and self.must_be_checked_x[x, y]:
            self.column_wall[x, y] = 1
            f = self.arrivable(self.Bx, self.By, 0) and self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.column_wall[x, y] = 0
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
            f = self.arrivable(self.Bx, self.By, 0) and self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.row_wall[x, y] = 0
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
            f = self.arrivable(self.Bx, self.By, 0) and self.arrivable(self.Wx, self.Wy, BOARD_LEN - 1)
            self.column_wall[x, y] = 0
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

    def dist_array(self, int goal_y, np.ndarray[DTYPE_t, ndim = 3] cross_arr):
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
        self.prev = np.zeros((BOARD_LEN, BOARD_LEN, 2), dtype="bool")
        return self.arrivable_(x, y, goal_y, isleft)

    def getroute(self, x, y, goal_y, isleft):
        self.seen = np.zeros((BOARD_LEN, BOARD_LEN), dtype="bool")
        self.prev = np.zeros((BOARD_LEN, BOARD_LEN, 2), dtype="bool")
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

    def calc_placable_array(self, skip_calc_graph=False):
        cdef np.ndarray[DTYPE_t, ndim = 3] cross_arr, cross_arr_copy, arr
        cdef np.ndarray[DTYPE_t, ndim = 2] row_array, column_array
        cdef int x, y
        cross_arr = self.cross_movable_array()
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
            cross_arr2[:, :, LEFT] = cross_arr[:, :, UP].T
            cross_arr2[:, :, DOWN] = cross_arr[:, :, RIGHT].T
            cross_arr2[:, :, RIGHT] = cross_arr[:, :, DOWN].T
            cross_arr2[:, :, UP] = cross_arr[:, :, LEFT].T
            column_cut1T = column_cut1.T
            column_cut2T = column_cut2.T
            column_array, self.column_special_cut, self.column_eq, self.column_graph = self.calc_placable_row_array(column_cut1T, column_cut2T, cross_arr2, self.By, self.Bx, self.Wy, self.Wx, False)
            column_array = column_array.T
            self.column_special_cut = self.column_special_cut.T

        self.must_be_checked_x = column_array & self.column_special_cut
        self.must_be_checked_y = row_array & self.row_special_cut

        for x in range(BOARD_LEN - 1):
            for y in range(BOARD_LEN - 1):
                f1, f2 = self.placable(x, y)
                row_array[x, y] = row_array[x, y] and f1
                column_array[x, y] = column_array[x, y] and f2
        return row_array, column_array

    def get_graph(self, np.ndarray[DTYPE_t, ndim = 3] cross_arr, int isgoaly):
        cdef int stack_p, i, dx, dy, xnext, ynext, x, y, x2, y2
        graph = {}
        cdef np.ndarray[DTYPE_t, ndim = 2] seen = np.zeros((BOARD_LEN, BOARD_LEN), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim = 1] stack = np.zeros((BOARD_LEN * BOARD_LEN), dtype=DTYPE)
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

    def calc_placable_row_array(self, np.ndarray[DTYPE_t, ndim = 2] row_cut1, np.ndarray[DTYPE_t, ndim = 2] row_cut2, np.ndarray[DTYPE_t, ndim = 3] cross_arr, int Bx, int By, int Wx, int Wy, int isgoaly):
        cdef int x, y, index, index2, id1, id2
        global total_time1, total_time2

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

        #print(row_cut1.T)
        #print(row_cut2)
        #print(edgess)
        #print(equivalence_class.T)
        #print(graph)
        return placable, special_row_cut2, equivalence_class, graph

    def get_all_cycles(self, np.ndarray[DTYPE_t, ndim = 3] cross_arr):
        cdef int x, y, i
        cdef np.ndarray[DTYPE_t, ndim = 3] direction_array
        cdef np.ndarray[DTYPE_t, ndim = 3] seen = np.zeros((BOARD_LEN, BOARD_LEN, 4), dtype=DTYPE)
        ret = []
        for x in range(BOARD_LEN):
            for y in range(BOARD_LEN):
                cross = cross_arr[x, y]
                for i in range(4):
                    if not seen[x, y, i] and cross[i] and not cross[(i - 1) % 4]:
                        direction_array = self.one_cycle(x, y, i, cross_arr)
                        ret.append(direction_array)
                        seen = seen | direction_array
        return ret

    def one_cycle(self, int x, int y, int first_direction, np.ndarray[DTYPE_t, ndim = 3] cross_arr):
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
            cross = cross_arr[x1, y1]
            index = directions[direction][0]
            while not cross[index]:
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
        prev = np.zeros((BOARD_LEN, BOARD_LEN, 2), dtype="int8")
        cross = np.zeros((4,), dtype="bool")
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
            cross[0] = (not (y == 0 or self.row_wall[min(x, BOARD_LEN - 2), y - 1] or self.row_wall[max(x - 1, 0), y - 1]))
            cross[1] = (not (x == BOARD_LEN - 1 or self.column_wall[x, min(y, BOARD_LEN - 2)] or self.column_wall[x, max(y - 1, 0)]))
            cross[2] = (not (y == BOARD_LEN - 1 or self.row_wall[min(x, BOARD_LEN - 2), y] or self.row_wall[max(x - 1, 0), y]))
            cross[3] = (not (x == 0 or self.column_wall[x - 1, min(y, BOARD_LEN - 2)] or self.column_wall[x - 1, max(y - 1, 0)]))
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
                    prev[x2, y2] = x, y
                    x_stack.append(x2)
                    y_stack.append(y2)
        #self.seen = seen
        self.prev = prev
        return False

    def display_cui(self):
        sys.stdout.write(" ")
        for c in ["a", "b", "c", "d", "e", "f", "g", "h", "i"]:
            sys.stdout.write(" " + c)
        print("")
        sys.stdout.write(" +")
        for x in range(BOARD_LEN):
            sys.stdout.write("-+")
        print("")

        for y in range(BOARD_LEN):
            sys.stdout.write(str(y + 1) + "|")
            for x in range(BOARD_LEN):
                if x == self.Bx and y == self.By:
                    sys.stdout.write("@")
                elif x == self.Wx and y == self.Wy:
                    sys.stdout.write("O")
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

        print("black walls:" + str(self.black_walls))
        print("white walls:" + str(self.white_walls))

        if self.turn % 2 == 0:
            print("{}:black turn".format(self.turn))
        else:
            print("{}:white turn".format(self.turn))
        print(total_time1, total_time2)
        self.check_placable_array_algo()

    def check_placable_array_algo(self):
        placabler1, placablec1 = self.old_calc_placable_array()
        placabler2, placablec2 = self.calc_placable_array()
        #print(np.all(placabler1 == placabler2) and np.all(placablec1 == placablec2))
        if not (np.all(placabler1 == placabler2) and np.all(placablec1 == placablec2)):
            print("")
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
            print("row")
            print(placabler1.T == placabler2.T)
            print(placabler1.T)
            print(placabler2.T)
            print("column")
            print(placablec1.T == placablec2.T)
            print(placablec1.T)
            print(placablec2.T)
            print("失敗")
            exit()

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
            temp = dist1
            dist1 = dist2
            dist2 = temp

        feature[Bx, By, 0] = 1.
        feature[Wx, Wy, 1] = 1.

        feature[:, :, 2] = black_walls / 10
        feature[:, :, 3] = white_walls / 10
        feature[:, :, 4] = turn % 2
        cdef np.ndarray[DTYPE_t, ndim = 3] cross_arr = self.cross_movable_array2(row_wall, column_wall)
        feature[:, :, 5:9] = cross_arr

        feature[:, :, 9] = dist1 / 20
        feature[:, :, 10] = dist2 / 20
        #if not xflip and not yflip:
        #    feature[:, :, 9] = self.dist_array1 / 10
        #    feature[:, :, 10] = self.dist_array2 / 10
        #else:
        #    feature[:, :, 9] = self.dist_array(0, cross_arr) / 10
        #    feature[:, :, 10] = self.dist_array(BOARD_LEN - 1, cross_arr) / 10
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

