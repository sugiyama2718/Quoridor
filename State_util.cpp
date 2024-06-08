#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include "State_util.h"
//#include <iostream>  //Could not find moduleが出る

__uint128_t public_cross_bitarrs[4];

void State_init(State* state) {
    state->row_wall_bitarr = state->column_wall_bitarr = 0;
    state->Bx = 4;
    state->By = 8;
    state->Wx = 4;
    state->Wy = 0;
    state->turn = 0;
    state->black_walls = state->white_walls = 10;

    calc_cross_bitarrs(state, state->row_wall_bitarr, state->column_wall_bitarr);

    for(int y = 0;y < BOARD_LEN;y++) {
        for(int x = 0;x < BOARD_LEN;x++) {
            state->dist_array1[x + y * BOARD_LEN] = y;
            state->dist_array2[x + y * BOARD_LEN] = BOARD_LEN - 1 - y;
        }
    }

    state->placable_r_bitarr = BIT_SMALL_BOARD_MASK;
    state->placable_c_bitarr = BIT_SMALL_BOARD_MASK;

    state->terminate = state->wall0_terminate = state->pseudo_terminate = false;
    state->reward = state->pseudo_reward = 0;
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

// __uint128_tをbool型の配列に変換する関数
bool boolArrayRet[128];
bool* uint128ToBoolArray(uint64_t value_high, uint64_t value_low) {
    __uint128_t value = ((__uint128_t)value_high << 64) | value_low;
    for (int i = 0; i < 128; ++i) {
        boolArrayRet[127 - i] = value & 1; // 最下位ビットを取り出す
        value >>= 1;              // 右に1ビットシフト
    }

    return boolArrayRet;
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

const int dxs[4] = {0, 1, 0, -1};
const int dys[4] = {-1, 0, 1, 0};

void movable_array(State* state, bool* mv, int x, int y, bool shortest_only=false) {
    // mvにmv[(dx + 1) + (dy + 1) * 3]の形で結果を格納
    uint8_t* dist_arr;
    int dx, dy, dx2, dy2, x2, y2, dx3, dy3;

    if(state->turn % 2 == 0) dist_arr = state->dist_array1;
    else dist_arr = state->dist_array2;

    for(int i = 0;i < 4;i++) {
        if(!get_bit(state->cross_bitarrs[i], x, y)) continue;
        dx = dxs[i];
        dy = dys[i];
        x2 = x + dx;
        y2 = y + dy;
        if((state->Bx == x2 && state->By == y2) || (state->Wx == x2 && state->Wy == y2)) {
            // 進む先にコマがある場合
            
            // 先に同じ方向に進むことができるかからチェック。進めるなら斜めには移動できない。
            if(get_bit(state->cross_bitarrs[i], x2, y2)) {
                if(shortest_only) {
                    if(dist_arr[(x2 + dx) + (y2 + dy) * BOARD_LEN] < dist_arr[x + y * BOARD_LEN]) mv[(dx + 1) + (dy + 1) * 3] = 1;
                } else {
                    mv[(dx + 1) + (dy + 1) * 3] = 1;
                }
                continue;
            }

            for(int j = 0;j < 4;j++) {
                dx2 = dxs[j];
                dy2 = dys[j];
                if(!get_bit(state->cross_bitarrs[j], x2, y2)) continue;
                if(shortest_only && (dist_arr[(x2 + dx2) + (y2 + dy2) * BOARD_LEN] >= dist_arr[x + y * BOARD_LEN])) continue;
                dx3 = max(min(dx + dx2, 1), -1);
                dy3 = max(min(dy + dy2, 1), -1);
                mv[(dx3 + 1) + (dy3 + 1) * 3] = 1;
            }
            mv[1 + 3] = 0;  // dx = dy = 0では0
        } else {
            // 進む先にコマがない場合
            if(shortest_only) {
                if(dist_arr[x2 + y2 * BOARD_LEN] < dist_arr[x + y * BOARD_LEN]) mv[(dx + 1) + (dy + 1) * 3] = 1;
            } else {
                mv[(dx + 1) + (dy + 1) * 3] = 1;
            }
            
        }
    }
}

bool accept_action_str(State* state, const char* s, bool check_placable=true, bool calc_placable_array=true, bool check_movable=true) {
    // 文字列sで表される行動を実行しようとし、その行動が合法手で実行できればtrue, そうでなければfalseを返す
    if(strlen(s) <= 1 || strlen(s) >= 4) return false;
    if(s[0] < 'a' && s[0] > 'i') return false;
    if(s[1] < '1' && s[1] > '9') return false;

    int x = s[0] - 'a', y = s[1] - '1';
    int x2, y2, dx, dy, x3, y3;
    bool mv[9] = {false}; // 配列の最初の要素をfalseで初期化し、残りの要素もfalseで初期化される
    int walls;
    int B_dist, W_dist;
    bool rf, cf;

    if(strlen(s) == 2) {
        //移動
        if(state->turn % 2 == 0) {
            x2 = state->Bx;
            y2 = state->By;
        } else {
            x2 = state->Wx;
            y2 = state->Wy;
        }
        dx = x - x2;
        dy = y - y2;
        if(std::abs(dx) + std::abs(dy) >= 3) {
            printf("dx + dy!!!!!!!\n");
            return false;
        }
        if(std::abs(dx) == 2 || std::abs(dy) == 2) {
            x3 = x2 + dx / 2;
            y3 = y2 + dy / 2;
            if(!((state->Bx == x3 && state->By == y3) || (state->Wx == x3 && state->Wy == y3))) return false;
            dx /= 2;
            dy /= 2;
        }
        if(check_movable) {
            movable_array(state, mv, x2, y2);
            if(!mv[(dx + 1) + (dy + 1) * 3]) {
                printf("%d %d\n", dx, dy);
                printf("not movable!!!!!!!\n");
                return false;
            }
        }
        if(state->turn % 2 == 0) {
            state->Bx = x;
            state->By = y;
        } else {
            state->Wx = x;
            state->Wy = y;
        }

        if(calc_placable_array) {
            calc_placable_array_and_set(state);
        }
    } else {
        //壁置き

        if(state->turn % 2 == 0) walls = state->black_walls;
        else walls = state->white_walls;

        rf = get_bit(state->placable_r_bitarr, x, y);
        cf = get_bit(state->placable_c_bitarr, x, y);

        if(check_placable) {
            if(s[2] == 'h') {
                if(rf && walls >= 1) {
                    set_row_wall_1(state, x, y);
                    if(state->turn % 2 == 0) state->black_walls -= 1;
                    else state->white_walls -= 1;
                } else {
                    return false;
                }
            } else if(s[2] == 'v') {
                if(cf && walls >= 1) {
                    set_column_wall_1(state, x, y);
                    if(state->turn % 2 == 0) state->black_walls -= 1;
                    else state->white_walls -= 1;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            if(s[2] == 'h') {
                set_row_wall_1(state, x, y);
                if(state->turn % 2 == 0) state->black_walls -= 1;
                else state->white_walls -= 1;
            } else if(s[2] == 'v') {
                set_column_wall_1(state, x, y);
                if(state->turn % 2 == 0) state->black_walls -= 1;
                else state->white_walls -= 1;
            } else {
                return false;
            }
        }

        if(calc_placable_array) {
            calc_placable_array_and_set(state);
        }

        calc_dist_array(state, 0);
        calc_dist_array(state, BOARD_LEN - 1);
    }
    state->turn++;

    //printf("%d %d\n", state->Bx, state->By);
    //printf("%d %d\n", state->Wx, state->Wy);

    if(state->By == 0) {
        state->terminate = true;
        state->reward = 1;
    } else if(state->Wy == BOARD_LEN - 1) {
        state->terminate = true;
        state->reward = -1;        
    } else if(state->turn == DRAW_TURN) {
        state->terminate = true;
        state->reward = 0;    
    }

    if((state->black_walls == 0 && state->white_walls == 0) || state->terminate) state->wall0_terminate = true;

    if(state->terminate) {
        state->pseudo_terminate = true;
        state->pseudo_reward = state->reward;
    } else {
        B_dist = state->dist_array1[state->Bx + state->By * BOARD_LEN];
        W_dist = state->dist_array2[state->Wx + state->Wy * BOARD_LEN];
        if(state->black_walls == 0 && (W_dist + (1 - state->turn % 2) <= B_dist - 1)) {
            state->pseudo_terminate = true;
            state->pseudo_reward = -1;
        } else if(state->white_walls == 0 && (B_dist + state->turn % 2 <= W_dist - 1)) {
            state->pseudo_terminate = true;
            state->pseudo_reward = 1;
        } else if(is_mirror_match(state)) {
            state->pseudo_terminate = true;
            state->pseudo_reward = -1;
        } else {
            state->pseudo_terminate = false;
            state->pseudo_reward = 0;
        }
    }

    return true;
}

int get_player1_dist_from_goal(State* state) {
    return state->dist_array1[state->Bx + state->By * BOARD_LEN];
}

int get_player2_dist_from_goal(State* state) {
    return state->dist_array2[state->Wx + state->Wy * BOARD_LEN];
}

__uint128_t flip_bitarr(__uint128_t bitarr) {
    // 8*8を縦軸・横軸両方でflipする

    __uint128_t mask, flip1 = 0, flip2 = 0;
    for(int x = 0;x < BOARD_LEN - 1;x++) {
        mask = SMALL_LEFT_EDGE >> x;
        if((BOARD_LEN - 2 * x - 2) >= 0) flip1 |= (bitarr & mask) >> (BOARD_LEN - 2 * x - 2);
        if((BOARD_LEN - 2 * x - 2) < 0) flip1 |= (bitarr & mask) << -(BOARD_LEN - 2 * x - 2);
    }
    for(int y = 0;y < BOARD_LEN - 1;y++) {
        mask = SMALL_UP_EDGE >> y * BIT_BOARD_LEN;
        if((BOARD_LEN - 2 * y - 2) >= 0) flip2 |= (flip1 & mask) >> (BOARD_LEN - 2 * y - 2) * BIT_BOARD_LEN;
        if((BOARD_LEN - 2 * y - 2) < 0) flip2 |= (flip1 & mask) << -(BOARD_LEN - 2 * y - 2) * BIT_BOARD_LEN;
    }
    return flip2;
}

bool is_mirror_match(State* state) {
    // 盤面上の壁が5枚以下ではmirror matchは成立し得ない
    if(20 - (state->black_walls + state->white_walls) <= 5) return false;

    if(state->black_walls != state->white_walls) return false;

    // コマが回転対称でなければmirror matchでない
    if(!(state->Bx == 8 - state->Wx && state->By == 8 - state->Wy)) return false;

    // 壁が回転対称でなければmirror matchでない
    if(!(state->row_wall_bitarr == flip_bitarr(state->row_wall_bitarr) && state->column_wall_bitarr == flip_bitarr(state->column_wall_bitarr))) return false;

    // 中央マスから横に移動できる場合、先手は横に移動することで優位に立てる可能性がある
    if(((state->row_wall_bitarr & CENTER_21_BOX) | (state->column_wall_bitarr & CENTER_21_BOX)) == 0) return false;

    // ゴールへの道が中央マスを必ず通る場合のみ後手勝利。
    __uint128_t blocked_cross_bitarr[4];
    bool B_arrivable, W_arrivable;
    for(int i = 0;i < 4;i++) blocked_cross_bitarr[i] = state->cross_bitarrs[i];
    if((state->column_wall_bitarr & CENTER_21_BOX) > 0) {
        set_0(&blocked_cross_bitarr[DOWN], 4, 3);
        set_0(&blocked_cross_bitarr[UP], 4, 4);
        set_0(&blocked_cross_bitarr[DOWN], 4, 4);
        set_0(&blocked_cross_bitarr[UP], 4, 5);
    } else {
        set_0(&blocked_cross_bitarr[RIGHT], 3, 4);
        set_0(&blocked_cross_bitarr[LEFT], 4, 4);
        set_0(&blocked_cross_bitarr[RIGHT], 4, 4);
        set_0(&blocked_cross_bitarr[LEFT], 5, 4);
    }
    B_arrivable = arrivable_by_cross(blocked_cross_bitarr, state->Bx, state->By, 0);
    W_arrivable = arrivable_by_cross(blocked_cross_bitarr, state->Wx, state->Wy, 0);

    if(B_arrivable && W_arrivable) return false;
    if(state->turn % 2 == 0 && B_arrivable) return false;
    if(state->turn % 2 == 1 && W_arrivable) return false;

    return true;
}

bool placable_r_with_color(State* state, int x, int y, int color) {
    if(get_bit(state->row_wall_bitarr, x, y) || get_bit(state->column_wall_bitarr, x, y)) return false;
    if(get_bit(state->row_wall_bitarr, max(x - 1, 0), y) || get_bit(state->row_wall_bitarr, min(x + 1, BOARD_LEN - 2), y)) return false;
    
    bool ret;
    
    set_row_wall_1(state, x, y);
    if(color == 0) {
        ret = arrivable_(state, state->Bx, state->By, 0);
    } else {
        ret = arrivable_(state, state->Wx, state->Wy, BOARD_LEN - 1);
    }
    set_row_wall_0(state, x, y);

    return ret;
}

bool placable_c_with_color(State* state, int x, int y, int color) {
    if(get_bit(state->row_wall_bitarr, x, y) || get_bit(state->column_wall_bitarr, x, y)) return false;
    if(get_bit(state->column_wall_bitarr, x, max(y - 1, 0)) || get_bit(state->column_wall_bitarr, x, min(y + 1, BOARD_LEN - 2))) return false;
    
    bool ret;
    
    set_column_wall_1(state, x, y);
    if(color == 0) {
        ret = arrivable_(state, state->Bx, state->By, 0);
    } else {
        ret = arrivable_(state, state->Wx, state->Wy, BOARD_LEN - 1);
    }
    set_column_wall_0(state, x, y);

    return ret;
}

__uint128_t calc_oneside_placable_r_cand_from_color(State* state, int color) {
    __uint128_t ret = 0;
    for(int y = 0; y < BOARD_LEN - 1; y++) {
        for(int x = 0; x < BOARD_LEN - 1; x++) {
            if(placable_r_with_color(state, x, y, color)) set_1(&ret, x, y);
        }
    }
    return ret;
}

__uint128_t calc_oneside_placable_c_cand_from_color(State* state, int color) {
    __uint128_t ret = 0;
    for(int y = 0; y < BOARD_LEN - 1; y++) {
        for(int x = 0; x < BOARD_LEN - 1; x++) {
            if(placable_c_with_color(state, x, y, color)) set_1(&ret, x, y);
        }
    }
    return ret;
}

bool is_certain_path_terminate(State* state, int color) {
    // color == -1のときは両方判定する
    int dist1 = get_player1_dist_from_goal(state);
    int dist2 = get_player2_dist_from_goal(state);
    __uint128_t placable_r, placable_c;
    uint8_t certain_dist_arr[81];
    int max_dist = -1, pawn_dist;

    // 先手後手それぞれで確定路があるか調べる
    if((color == -1 || color == 0) && (dist1 + state->turn % 2 <= dist2 - 1)) {  // dist1 <= certain_dist1なのでこれを満たさないときは判定不要
        placable_r = calc_oneside_placable_r_cand_from_color(state, 0);
        placable_c = calc_oneside_placable_c_cand_from_color(state, 0);
        calc_cross_bitarrs_global(state->row_wall_bitarr | placable_r, state->column_wall_bitarr | placable_c);
        calc_dist_array_inner(certain_dist_arr, 0, public_cross_bitarrs);
        for(int i = 0;i < 81;i++) {
            if(certain_dist_arr[i] > max_dist) max_dist = certain_dist_arr[i];
        }
        pawn_dist = certain_dist_arr[state->Bx + state->By * BOARD_LEN];
        if(max_dist == pawn_dist) return false;  // 到達不能
        if(pawn_dist + state->turn % 2 <= dist2 - 1) return true;
    }

    if((color == -1 || color == 1) && (dist2 + (1 - state->turn % 2) <= dist1 - 1)) { 
        placable_r = calc_oneside_placable_r_cand_from_color(state, 1);
        placable_c = calc_oneside_placable_c_cand_from_color(state, 1);
        calc_cross_bitarrs_global(state->row_wall_bitarr | placable_r, state->column_wall_bitarr | placable_c);
        calc_dist_array_inner(certain_dist_arr, BOARD_LEN - 1, public_cross_bitarrs);
        for(int i = 0;i < 81;i++) {
            if(certain_dist_arr[i] > max_dist) max_dist = certain_dist_arr[i];
        }
        pawn_dist = certain_dist_arr[state->Wx + state->Wy * BOARD_LEN];
        if(max_dist == pawn_dist) return false;  // 到達不能
        if(pawn_dist + (1 - state->turn % 2) <= dist1 - 1) return true;
    }

    return false;
}

BitArrayPair placable_array(State* state, int color) {
    // 壁の枚数も考慮した壁に関する合法手配列を返す
    BitArrayPair ret;
    if((color == 0 && state->black_walls == 0) || (color == 1 && state->white_walls == 0)) {
        ret.bitarr1 = 0;
        ret.bitarr2 = 0;
    } else {
        ret = calc_placable_array_(state);
    }
    return ret;
}

void calc_cross_bitarrs_global(__uint128_t row_bitarr, __uint128_t column_bitarr) {
    public_cross_bitarrs[UP] = UP_EDGE;
    public_cross_bitarrs[UP] |= down_shift(row_bitarr);
    public_cross_bitarrs[UP] |= right_down_shift(row_bitarr);

    public_cross_bitarrs[RIGHT] = RIGHT_EDGE;
    public_cross_bitarrs[RIGHT] |= column_bitarr;
    public_cross_bitarrs[RIGHT] |= down_shift(column_bitarr);

    public_cross_bitarrs[DOWN] = DOWN_EDGE;
    public_cross_bitarrs[DOWN] |= row_bitarr;
    public_cross_bitarrs[DOWN] |= right_shift(row_bitarr);

    public_cross_bitarrs[LEFT] = LEFT_EDGE;
    public_cross_bitarrs[LEFT] |= right_shift(column_bitarr);
    public_cross_bitarrs[LEFT] |= right_down_shift(column_bitarr);

    for(int i = 0; i < 4; i++) {
        public_cross_bitarrs[i] = ~public_cross_bitarrs[i];
        public_cross_bitarrs[i] &= BIT_BOARD_MASK;
    }
}

void calc_cross_bitarrs(State* state, __uint128_t row_bitarr, __uint128_t column_bitarr) {
    calc_cross_bitarrs_global(row_bitarr, column_bitarr);

    for(int i = 0; i < 4; i++) {
        state->cross_bitarrs[i] = public_cross_bitarrs[i];
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

void calc_dist_array_inner(uint8_t* dist_arr_p, int goal_y, __uint128_t cross_bitarrs[4]) {
    // dist_arr_pに格納
    Point_uint8 point_queue[BOARD_LEN * BOARD_LEN];
    int q_s = 0, q_e = 0;
    int x2, y2, x3, y3, dx, dy;
    uint8_t max_dist = 0;
    static const int dxs[4] = {0, 1, 0, -1};
    static const int dys[4] = {-1, 0, 1, 0};

    for(int i = 0;i < BOARD_LEN * BOARD_LEN;i++) dist_arr_p[i] = 0xFF;

    for(int x = 0;x < BOARD_LEN;x++) {
        dist_arr_p[x + goal_y * BOARD_LEN] = 0;
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
            get_bit(cross_bitarrs[i], x2, y2) && (dist_arr_p[x3 + y3 * BOARD_LEN] > dist_arr_p[x2 + y2 * BOARD_LEN] + 1)) {
                max_dist = dist_arr_p[x3 + y3 * BOARD_LEN] = dist_arr_p[x2 + y2 * BOARD_LEN] + 1;
                point_queue[q_e].x = x3;
                point_queue[q_e].y = y3;
                q_e++;
            }
        }
    }

    max_dist++;

    for(int i = 0;i < BOARD_LEN * BOARD_LEN;i++) {
        if(dist_arr_p[i] == 0xFF) dist_arr_p[i] = max_dist;
    }
}

void calc_dist_array(State* state, int goal_y) {
    // goal_yの値から判断して結果をstateのdist_arrayに格納
    uint8_t* dist_arr_p;
    if(goal_y == 0) dist_arr_p = state->dist_array1;
    else dist_arr_p = state->dist_array2;

    calc_dist_array_inner(dist_arr_p, goal_y, state->cross_bitarrs);
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

void calc_placable_array_and_set(State* state) {
    BitArrayPair pair = calc_placable_array_(state);
    state->placable_r_bitarr = pair.bitarr1;
    state->placable_c_bitarr = pair.bitarr2;
}
