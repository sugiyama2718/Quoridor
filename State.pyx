# coding:utf-8
import numpy as np
cimport numpy as np
import sys
import time

DTYPE = np.int
ctypedef np.int_t DTYPE_t

BOARD_LEN = 9
DRAW_TURN = 200
notation_dict = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8}
total_time1 = 0.
total_time2 = 0.

cdef class State:
    draw_turn = DRAW_TURN

    cdef public np.ndarray seen, row_wall, column_wall, must_be_checked_x, must_be_checked_y, prev, placable_rb, placable_cb, placable_rw, placable_cw
    cdef public int Bx, By, Wx, Wy, turn, black_walls, white_walls, terminate, reward
    def __init__(self):
        self.row_wall = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.column_wall = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype="bool")
        self.placable_rb = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        self.placable_cb = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        self.placable_rw = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        self.placable_cw = np.ones((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        self.must_be_checked_x = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        self.must_be_checked_y = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        self.Bx = BOARD_LEN // 2
        self.By = BOARD_LEN - 1
        self.Wx = BOARD_LEN // 2
        self.Wy = 0
        self.turn = 0
        self.black_walls = 10
        self.white_walls = 10
        self.terminate = False
        self.reward = 0  # blackから見たreward
        self.seen = np.zeros((BOARD_LEN, BOARD_LEN), dtype=DTYPE)

    def __eq__(self, state):
        f = np.all(self.row_wall == state.row_wall) and np.all(self.column_wall == state.column_wall)
        f = f and self.Bx == state.Bx and self.By == state.By and self.Wx == state.Wx and self.Wy == state.Wy
        f = f and self.black_walls == state.black_walls and self.white_walls == state.white_walls
        return f


    def color_p(self, color):
        if color == 0:
            return self.Bx, self.By
        else:
            return self.Wx, self.Wy

    def accept_action_str(self, s):
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
            points = []
            mendoi = False
            if abs(dx) == 1 and dy == 0:
                dx = (dx + 2) // 2 - 1
                points.append((x0 + dx, y0))
                points.append((x0 + dx, y0 - 1))
                if jump:
                    if dx == -1:
                        dx = -2
                    elif dx == 0:
                        dx = 1
                    points.append((x0 + dx, y0))
                    points.append((x0 + dx, y0 - 1))
            elif abs(dy) == 1 and dx == 0:
                dy = (dy + 2) // 2 - 1
                points.append((x0, y0 + dy))
                points.append((x0 - 1, y0 + dy))
                if jump:
                    if dy == -1:
                        dy = -2
                    elif dy == 0:
                        dy = 1
                    points.append((x0, y0 + dy))
                    points.append((x0 - 1, y0 + dy))
            else:
                # 斜め移動の場合めんどいので計算しなおし
                mendoi = True
            if jump:
                mendoi = True

            if mendoi:
                placable_r, placable_c = self.calc_placable_array()
                self.placable_rb, self.placable_cb = (placable_r * (self.black_walls >= 1), placable_c * self.black_walls >= 1)
                self.placable_rw, self.placable_cw = (placable_r * (self.white_walls >= 1), placable_c * self.white_walls >= 1)
            else:
                self.must_be_checked_x = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
                self.must_be_checked_y = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
                for x, y in points:
                    if not self.inboard(x, y, BOARD_LEN - 1):
                        continue
                    self.must_be_checked_x[x, y] = True
                    self.must_be_checked_y[x, y] = True
                    rf, cf = self.placable(x, y)
                    self.placable_rb[x, y], self.placable_cb[x, y] = (rf and self.black_walls >= 1, cf and self.black_walls >= 1)
                    self.placable_rw[x, y], self.placable_cw[x, y] = (rf and self.white_walls >= 1, cf and self.white_walls >= 1)

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
    def cross_movable(self, int x, int y):
        ret = np.zeros((4,), dtype=DTYPE)
        ret[0] = (not (y == 0 or self.row_wall[min(x, BOARD_LEN - 2), y - 1] or self.row_wall[max(x - 1, 0), y - 1]))
        ret[1] = (not (x == BOARD_LEN - 1 or self.column_wall[x, min(y, BOARD_LEN - 2)] or self.column_wall[x, max(y - 1, 0)]))
        ret[2] = (not (y == BOARD_LEN - 1 or self.row_wall[min(x, BOARD_LEN - 2), y] or self.row_wall[max(x - 1, 0), y]))
        ret[3] = (not (x == 0 or self.column_wall[x - 1, min(y, BOARD_LEN - 2)] or self.column_wall[x - 1, max(y - 1, 0)]))
        return ret

    def movable_array(self, x, y):
        cdef np.ndarray[DTYPE_t, ndim=1] cross, cross2
        mv = np.zeros((3, 3), dtype=bool)
        cross = self.cross_movable(x, y)
        for i, p in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            if not cross[i]:
                continue
            x2 = x + p[0]
            y2 = y + p[1]
            if (self.Bx == x2 and self.By == y2) or (self.Wx == x2 and self.Wy == y2):
                cross2 = self.cross_movable(x2, y2)
                movable_list = []
                for j, q in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
                    if cross2[j]:
                        movable_list.append((max(min(p[0] + q[0], 1), -1), max(min(p[1] + q[1], 1), -1)))
                for movable_p in movable_list:
                    mv[movable_p] = 1
                mv[0, 0] = 0
            else:
                mv[p] = 1
        return mv

    def inboard(self, int x, int y, int size):
        if x < 0 or y < 0 or x >= size or y >= size:
            return False
        return True

    def placable_array(self, int color):
        if color == 0:
            return self.placable_rb, self.placable_cb
        else:
            return self.placable_rw, self.placable_cw

    def calc_placable_array(self):
        cdef np.ndarray[DTYPE_t, ndim=2] ret1 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] ret2 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] must_be_checked_x1 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] must_be_checked_x2 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] must_be_checked_y1 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] must_be_checked_y2 = np.zeros((BOARD_LEN - 1, BOARD_LEN - 1), dtype=DTYPE)
        global total_time1, total_time2
        start = time.time()
        routes = [self.getroute(self.Bx, self.By, 0, True), self.getroute(self.Bx, self.By, 0, False)]

        total_time1 += time.time() - start
        start = time.time()

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

        total_time2 += time.time() - start

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

    def placable(self, int x, int y):
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

    def placable_r(self, int x, int y):
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

    def placable_c(self, int x, int y):
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
        cdef np.ndarray[DTYPE_t, ndim=1] cross
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

    def arrivable(self, int x, int y, int goal_y, isleft=True):
        self.seen = np.zeros((BOARD_LEN, BOARD_LEN), dtype=DTYPE)
        self.prev = np.zeros((BOARD_LEN, BOARD_LEN, 2), dtype=DTYPE)
        return self.arrivable_(x, y, goal_y, isleft)

    def getroute(self, int x, int y, int goal_y, int isleft):
        cdef int x2, y2
        self.seen = np.zeros((BOARD_LEN, BOARD_LEN), dtype=DTYPE)
        self.prev = np.zeros((BOARD_LEN, BOARD_LEN, 2), dtype=DTYPE)
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


    """
    def getroute(self, x, y, goal_y, isleft):
        prev = -np.ones((BOARD_LEN, BOARD_LEN, 2), dtype=DTYPE)

        if isleft and goal_y == 0:
            directions = [(3, -1, 0), (0, 0, -1), (1, 1, 0), (2, 0, 1)]
        elif isleft and goal_y == BOARD_LEN - 1:
            directions = [(1, 1, 0), (2, 0, 1), (3, -1, 0), (0, 0, -1)]
        elif not isleft and goal_y == 0:
            directions = [(1, 1, 0), (0, 0, -1), (3, -1, 0), (2, 0, 1)]
        elif not isleft and goal_y == BOARD_LEN - 1:
            directions = [(3, -1, 0), (2, 0, 1), (1, 1, 0), (0, 0, -1)]

        x1 = x
        y1 = y
        direction = 0
        self.display_cui()
        print(x, y, goal_y, isleft)
        isstart = True
        while True:
            if y1 == goal_y:
                break
            cross = self.cross_movable(x1, y1)
            index = directions[direction][0]
            while not cross[index]:
                direction = (direction + 1) % 4
                index = directions[direction][0]
                isstart = False
            x2 = x1 + directions[direction][1]
            y2 = y1 + directions[direction][2]

            if prev[x2, y2][0] == -1 and prev[x2, y2][1] == -1:
                prev[x2, y2] = x1, y1
            x1 = x2
            y1 = y2

            if not isstart:
                direction = (direction - 1) % 4

        x2 = np.argmax(prev[:, goal_y, 1])
        y2 = goal_y
        route = []
        while not (x2 == x and y2 == y):
            route.append((x2, y2))
            x2, y2 = prev[x2, y2]
        route.append((x, y))
        route = route[::-1]
        return route
    """


    def arrivable_(self, int x, int y, int goal_y, int isleft):
        cdef int i, x2, y2, dx, dy
        #cdef np.ndarray[DTYPE_t, ndim=2] seen
        #cdef np.ndarray[DTYPE_t, ndim=1] cross
        #cdef np.ndarray[DTYPE_t, ndim=3] prev
        #seen = np.zeros((BOARD_LEN, BOARD_LEN), dtype=DTYPE)
        prev = np.zeros((BOARD_LEN, BOARD_LEN, 2), dtype=DTYPE)
        cross = np.zeros((4,), dtype=DTYPE)
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
        print("{} {}".format(total_time1, total_time2))

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

