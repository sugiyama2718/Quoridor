#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <cmath>
//#include <iostream>  //Could not find moduleが出る

extern "C" {

// State中の__uint128_tの変数はすべてbitarray
// ここを変更したらpythonの呼び出し側の定義も必ず変更すること！！！！！！
struct State {
    __uint128_t row_wall_bitarr, column_wall_bitarr;
    __uint128_t cross_bitarrs[4];
    int Bx, By, Wx, Wy;  // Bが先手、Wが後手。BGAと逆になっているが、昔実装したときに混同した名残。
    int turn;
    int black_walls, white_walls;
};

const int BOARD_LEN = 9;
const int BIT_BOARD_LEN = 11;
const int ACTION_NUM = 137;
enum DIRECTION {
    UP,
    RIGHT,
    DOWN,
    LEFT
};
const __uint128_t BIT_BOARD_MASK = ((__uint128_t)0xFF9FF3FE7FCFF9FFULL << 64) | 0x3FE7FCFF80000000ULL;  // 9*9
const __uint128_t BIT_SMALL_BOARD_MASK = ((__uint128_t)0xFF1FE3FC7F8FF1FEULL << 64) | 0x3FC7F80000000000ULL;  // 8*8
const __uint128_t UP_EDGE = ((__uint128_t)0xFF80000000000000ULL << 64) | 0x0000000000000000ULL;
const __uint128_t RIGHT_EDGE = ((__uint128_t)0x80100200400801ULL << 64) | 0x0020040080000000ULL;
const __uint128_t DOWN_EDGE = 0xFF80000000ULL;
const __uint128_t LEFT_EDGE = ((__uint128_t)0x8010020040080100ULL << 64) | 0x2004008000000000ULL;
const __uint128_t BOX_10 = ((__uint128_t)0XFFD00A0140280500ULL << 64) | 0xA01402805FF80000ULL;

void print_bitarray(__uint128_t bitarr);
void print_full_bitarray(__uint128_t bitarr);
inline __uint128_t up_shift(__uint128_t bitarr);
inline __uint128_t right_shift(__uint128_t bitarr);
inline __uint128_t down_shift(__uint128_t bitarr);
inline __uint128_t left_shift(__uint128_t bitarr);
inline __uint128_t right_down_shift(__uint128_t bitarr);
int arrivable_(State* state, int pawn_x, int pawn_y, int goal_y);
void calc_cross_bitarrs(State* state, __uint128_t row_bitarr, __uint128_t column_bitarr);

struct BitArrayPair {
    __uint128_t bitarr1;
    __uint128_t bitarr2;
};

struct Point_uint8 {
    uint8_t x, y;
};

struct Point_int {
    int x, y;
};

void State_init(State* state) {
    state->row_wall_bitarr = state->column_wall_bitarr = 0;
    state->Bx = 4;
    state->By = 8;
    state->Wx = 4;
    state->Wy = 0;
    state->turn = 0;
    state->black_walls = state->white_walls = 10;
    calc_cross_bitarrs(state, state->row_wall_bitarr, state->column_wall_bitarr);
}

bool eq_state(State* state1, State* state2) {
    return (state1->row_wall_bitarr == state2->row_wall_bitarr) && (state1->column_wall_bitarr == state2->column_wall_bitarr)
    && (state1->Bx == state2->Bx) && (state1->By == state2->By) && (state1->Wx == state2->Wx) && (state1->Wy == state2->Wy)
    && (state1->turn == state2->turn)
    && (state1->black_walls == state2->black_walls) && (state1->white_walls == state2->white_walls);
}

Point_int color_p(State* state, int color) {
    Point_int ret;
    if(color == 0) {
        ret.x = state->Bx;
        ret.y = state->By;
    } else {
        ret.x = state->Wx;
        ret.y = state->Wy;
    }
    return ret;
}

void copy_state(State* new_state, State* old_state) {
    *new_state = *old_state;
}

inline bool get_bit(__uint128_t bitarr, int x, int y) {
    return (bitarr & ((__uint128_t)1 << (127 - (x + y * BIT_BOARD_LEN)))) > 0;
}

void set_0(__uint128_t* bitarr, int x, int y) {
    *bitarr &= ~((__uint128_t)1 << (127 - (x + y * BIT_BOARD_LEN)));
}

void set_1(__uint128_t* bitarr, int x, int y) {
    *bitarr |= (__uint128_t)1 << (127 - (x + y * BIT_BOARD_LEN));
}

void set_row_wall_1(State* state, int x, int y) {
    set_1(&state->row_wall_bitarr, x, y);
    calc_cross_bitarrs(state, state->row_wall_bitarr, state->column_wall_bitarr);
}
void set_row_wall_0(State* state, int x, int y) {
    set_0(&state->row_wall_bitarr, x, y);
    calc_cross_bitarrs(state, state->row_wall_bitarr, state->column_wall_bitarr);
}
void set_column_wall_1(State* state, int x, int y) {
    set_1(&state->column_wall_bitarr, x, y);
    calc_cross_bitarrs(state, state->row_wall_bitarr, state->column_wall_bitarr);
}
void set_column_wall_0(State* state, int x, int y) {
    set_0(&state->column_wall_bitarr, x, y);
    calc_cross_bitarrs(state, state->row_wall_bitarr, state->column_wall_bitarr);
}

void print_bitarray(__uint128_t bitarr) {
    for(int y = 0; y < BOARD_LEN; y++) {
        for(int x = 0; x < BOARD_LEN; x++) {
            __uint128_t shifted = (bitarr << (x + y * BIT_BOARD_LEN)) >> 64;
            printf("%d", (uint64_t)(shifted & 0x8000000000000000) >> 63);
        }
        printf("\n");
    }
}

void print_state(State* state) {
    printf("row_wall\n");
    print_bitarray(state->row_wall_bitarr);
    printf("column_wall\n");
    print_bitarray(state->column_wall_bitarr);
    printf("Bx, By, Wx, Wy = %d, %d, %d, %d\n", state->Bx, state->By, state->Wx, state->Wy);
    printf("turn=%d\n", state->turn);
    printf("black_walls=%d, white_walls=%d\n", state->black_walls, state->white_walls);
}

void print_full_bitarray(__uint128_t bitarr) {
    for(int y = 0; y < BIT_BOARD_LEN; y++) {
        for(int x = 0; x < BIT_BOARD_LEN; x++) {
            __uint128_t shifted = (bitarr << (x + y * BIT_BOARD_LEN)) >> 64;
            printf("%d", (uint64_t)(shifted & 0x8000000000000000) >> 63);
        }
        printf("\n");
    }
    printf("last=%d\n", (uint64_t)(bitarr & 0xFFFFFFFFFFFFFFFF) & 0x7F);
}

inline __uint128_t up_shift(__uint128_t bitarr) {
    return bitarr << BIT_BOARD_LEN;
}

inline __uint128_t right_shift(__uint128_t bitarr) {
    return bitarr >> 1;
}

inline __uint128_t down_shift(__uint128_t bitarr) {
    return bitarr >> BIT_BOARD_LEN;
}

inline __uint128_t left_shift(__uint128_t bitarr) {
    return bitarr << 1;
}

inline __uint128_t left_up_shift(__uint128_t bitarr) {
    return bitarr << (BIT_BOARD_LEN + 1);
}

inline __uint128_t right_down_shift(__uint128_t bitarr) {
    return bitarr >> (BIT_BOARD_LEN + 1);
}

inline __uint128_t right_right_down_shift(__uint128_t bitarr) {
    return bitarr >> (BIT_BOARD_LEN + 2);
}

inline __uint128_t right_down_down_shift(__uint128_t bitarr) {
    return bitarr >> (BIT_BOARD_LEN * 2 + 1);
}

int select_action(float Q[ACTION_NUM], float N[ACTION_NUM], float P[ACTION_NUM],
float C_puct, float estimated_V, int color, int turn) {
    int N_sum = 0;
    for(int i = 0;i < ACTION_NUM;i++) {
        N_sum += N[i];
    }
    float N_sum_sqrt = std::sqrt(1 + N_sum);

    int a = -1;
    float x_max = -2.0;
    float Qi, x;
    for(int i = 0;i < ACTION_NUM;i++) {
        if(P[i] == 0.0) continue;

        if(N[i] == 0) Qi = estimated_V;
        else Qi = Q[i];

        if(color == turn % 2) x = Qi;
        else x = -Qi;

        x = x + C_puct * P[i] * N_sum_sqrt / (1 + N[i]);
        if(x > x_max) {
            a = i;
            x_max = x;
        }
    }
    
    return a;
}

void calc_cross_bitarrs(State* state, __uint128_t row_bitarr, __uint128_t column_bitarr) {
    state->cross_bitarrs[UP] = UP_EDGE;
    state->cross_bitarrs[UP] |= down_shift(row_bitarr);
    state->cross_bitarrs[UP] |= right_down_shift(row_bitarr);

    state->cross_bitarrs[RIGHT] = RIGHT_EDGE;
    state->cross_bitarrs[RIGHT] |= column_bitarr;
    state->cross_bitarrs[RIGHT] |= down_shift(column_bitarr);

    state->cross_bitarrs[DOWN] = DOWN_EDGE;
    state->cross_bitarrs[DOWN] |= row_bitarr;
    state->cross_bitarrs[DOWN] |= right_shift(row_bitarr);

    state->cross_bitarrs[LEFT] = LEFT_EDGE;
    state->cross_bitarrs[LEFT] |= right_shift(column_bitarr);
    state->cross_bitarrs[LEFT] |= right_down_shift(column_bitarr);

    for(int i = 0; i < 4; i++) {
        state->cross_bitarrs[i] = ~state->cross_bitarrs[i];
        state->cross_bitarrs[i] &= BIT_BOARD_MASK;
    }
}

int arrivable_by_cross(__uint128_t cross_bitarrs[4], int pawn_x, int pawn_y, int goal_y) {
    __uint128_t seen_bitarr = ((__uint128_t)1 << (127 - (pawn_x + pawn_y * BIT_BOARD_LEN)));
    __uint128_t seen_bitarr_prev = ((__uint128_t)1 << (127 - (pawn_x + pawn_y * BIT_BOARD_LEN)));

    do {
        seen_bitarr_prev = seen_bitarr;
        seen_bitarr |= up_shift(seen_bitarr_prev & cross_bitarrs[UP]);
        seen_bitarr |= right_shift(seen_bitarr_prev & cross_bitarrs[RIGHT]);
        seen_bitarr |= down_shift(seen_bitarr_prev & cross_bitarrs[DOWN]);
        seen_bitarr |= left_shift(seen_bitarr_prev & cross_bitarrs[LEFT]);
        seen_bitarr &= BIT_BOARD_MASK;

        if(goal_y == 0) {
            if((seen_bitarr & UP_EDGE) > 0) return true;
        } else {
            if((seen_bitarr & DOWN_EDGE) > 0) return true;
        }
    } while(seen_bitarr != seen_bitarr_prev);

    return false;
}

uint8_t dist_array_ret[BOARD_LEN * BOARD_LEN];

uint8_t* calc_dist_array(State* state, int goal_y) {
    Point_uint8 point_queue[BOARD_LEN * BOARD_LEN];
    int q_s = 0, q_e = 0;
    int x2, y2, x3, y3, dx, dy;
    uint8_t max_dist = 0;
    static const int dxs[4] = {0, 1, 0, -1};
    static const int dys[4] = {-1, 0, 1, 0};

    for(int i = 0;i < BOARD_LEN * BOARD_LEN;i++) dist_array_ret[i] = 0xFF;

    for(int x = 0;x < BOARD_LEN;x++) {
        dist_array_ret[x + goal_y * BOARD_LEN] = 0;
        point_queue[q_e].x = x;
        point_queue[q_e].y = goal_y;
        q_e++;
    }

    while(q_e - q_s > 0) {
        x2 = point_queue[q_s].x;
        y2 = point_queue[q_s].y;
        q_s++;
        for(int i = 0;i < 4;i++) {
            dx = dxs[i];
            dy = dys[i];
            x3 = x2 + dx;
            y3 = y2 + dy;

            if(!(x3 < 0 || x3 >= BOARD_LEN || y3 < 0 || y3 >= BOARD_LEN) && 
            get_bit(state->cross_bitarrs[i], x2, y2) && (dist_array_ret[x3 + y3 * BOARD_LEN] > dist_array_ret[x2 + y2 * BOARD_LEN] + 1)) {
                max_dist = dist_array_ret[x3 + y3 * BOARD_LEN] = dist_array_ret[x2 + y2 * BOARD_LEN] + 1;
                point_queue[q_e].x = x3;
                point_queue[q_e].y = y3;
                q_e++;
            }
        }
    }

    max_dist++;

    for(int i = 0;i < BOARD_LEN * BOARD_LEN;i++) {
        if(dist_array_ret[i] == 0xFF) dist_array_ret[i] = max_dist;
    }

    return dist_array_ret;
}

int arrivable_(State* state, int pawn_x, int pawn_y, int goal_y) {
    return arrivable_by_cross(state->cross_bitarrs, pawn_x, pawn_y, goal_y);
}

BitArrayPair calc_placable_array_(State* state) {

    // 各交点について壁に触れているなら1が立っているbitarrayを作る
    __uint128_t wall_point_bitarr = BOX_10;
    wall_point_bitarr |= down_shift(state->row_wall_bitarr);
    wall_point_bitarr |= right_down_shift(state->row_wall_bitarr);
    wall_point_bitarr |= right_right_down_shift(state->row_wall_bitarr);
    wall_point_bitarr |= right_shift(state->column_wall_bitarr);
    wall_point_bitarr |= right_down_shift(state->column_wall_bitarr);
    wall_point_bitarr |= right_down_down_shift(state->column_wall_bitarr);

    // arrivableを使って壁が置けるか判定する必要がある箇所を求める
    __uint128_t must_be_checked_x = (up_shift(wall_point_bitarr) & wall_point_bitarr) | (wall_point_bitarr & down_shift(wall_point_bitarr)) | (up_shift(wall_point_bitarr) & down_shift(wall_point_bitarr));
    __uint128_t must_be_checked_y = (left_shift(wall_point_bitarr) & wall_point_bitarr) | (wall_point_bitarr & right_shift(wall_point_bitarr)) | (left_shift(wall_point_bitarr) & right_shift(wall_point_bitarr));
    must_be_checked_x &= BIT_BOARD_MASK;
    must_be_checked_y &= BIT_BOARD_MASK;
    must_be_checked_x = left_up_shift(must_be_checked_x);
    must_be_checked_y = left_up_shift(must_be_checked_y);
    must_be_checked_x &= BIT_BOARD_MASK;
    must_be_checked_y &= BIT_BOARD_MASK;

    BitArrayPair ret;
    ret.bitarr1 = ~(state->row_wall_bitarr | state->column_wall_bitarr | left_shift(state->row_wall_bitarr) | right_shift(state->row_wall_bitarr));
    ret.bitarr2 = ~(state->row_wall_bitarr | state->column_wall_bitarr | up_shift(state->column_wall_bitarr) | down_shift(state->column_wall_bitarr));
    ret.bitarr1 &= BIT_SMALL_BOARD_MASK;
    ret.bitarr2 &= BIT_SMALL_BOARD_MASK;

    // 既においてある壁とぶつかるのを除外
    must_be_checked_x &= ret.bitarr2;
    must_be_checked_y &= ret.bitarr1;

    // print_full_bitarray(state->row_wall_bitarr);
    // print_full_bitarray(state->column_wall_bitarr);
    // print_full_bitarray(BIT_SMALL_BOARD_MASK);
    // print_bitarray(ret.bitarr1);

    __uint128_t virtual_row_wall, virtual_column_wall;
    for(int y = 0;y < BOARD_LEN - 1;y++) {
        for(int x = 0;x < BOARD_LEN - 1;x++) {
            virtual_row_wall = ((__uint128_t)1 << (127 - (x + y * BIT_BOARD_LEN))) & must_be_checked_y;
            if(virtual_row_wall > 0) {
                set_row_wall_1(state, x, y);
                if(!arrivable_by_cross(state->cross_bitarrs, state->Bx, state->By, 0)
                || !arrivable_by_cross(state->cross_bitarrs, state->Wx, state->Wy, BOARD_LEN - 1)) ret.bitarr1 &= ~((__uint128_t)1 << (127 - (x + y * BIT_BOARD_LEN)));
                set_row_wall_0(state, x, y);
            }

            virtual_column_wall = ((__uint128_t)1 << (127 - (x + y * BIT_BOARD_LEN))) & must_be_checked_x;
            if(virtual_column_wall > 0) {
                set_column_wall_1(state, x, y);
                if(!arrivable_by_cross(state->cross_bitarrs, state->Bx, state->By, 0)
                || !arrivable_by_cross(state->cross_bitarrs, state->Wx, state->Wy, BOARD_LEN - 1)) ret.bitarr2 &= ~((__uint128_t)1 << (127 - (x + y * BIT_BOARD_LEN)));
                set_column_wall_0(state, x, y);
            }
        }
    }

    return ret;
}

}
