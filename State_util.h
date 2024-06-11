#include <cinttypes>

#ifndef STATE_UTIL_H
#define STATE_UTIL_H

extern "C" {

// State中の__uint128_tの変数はすべてbitarray
// ここを変更したらpythonの呼び出し側の定義も必ず変更すること！！！！！！
struct State {
    __uint128_t row_wall_bitarr, column_wall_bitarr;
    __uint128_t cross_bitarrs[4];
    int Bx, By, Wx, Wy;  // Bが先手、Wが後手。BGAと逆になっているが、昔実装したときに混同した名残。
    int turn;
    int black_walls, white_walls;
    uint8_t dist_array1[81];  // x + y * BOARD_LENでアクセスするものとする。1が先手、2は後手
    uint8_t dist_array2[81];
    __uint128_t placable_r_bitarr, placable_c_bitarr;
    bool terminate, wall0_terminate, pseudo_terminate;
    int reward, pseudo_reward;
};

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

const int BOARD_LEN = 9;
const int BIT_BOARD_LEN = 11;
const int ACTION_NUM = 137;
const int DRAW_TURN = 300;
enum DIRECTION {
    UP,
    RIGHT,
    DOWN,
    LEFT
};
const __uint128_t BIT_BOARD_MASK = ((__uint128_t)0xFF9FF3FE7FCFF9FFULL << 64) | 0x3FE7FCFF80000000ULL;  // 9*9
const __uint128_t BIT_SMALL_BOARD_MASK = ((__uint128_t)0xFF1FE3FC7F8FF1FEULL << 64) | 0x3FC7F80000000000ULL;  // 8*8

// len 9
const __uint128_t UP_EDGE = ((__uint128_t)0xFF80000000000000ULL << 64) | 0x0000000000000000ULL;
const __uint128_t RIGHT_EDGE = ((__uint128_t)0x80100200400801ULL << 64) | 0x0020040080000000ULL;
const __uint128_t DOWN_EDGE = 0xFF80000000ULL;
const __uint128_t LEFT_EDGE = ((__uint128_t)0x8010020040080100ULL << 64) | 0x2004008000000000ULL;

// len 8
const __uint128_t SMALL_UP_EDGE = ((__uint128_t)0xFF00000000000000ULL << 64) | 0x0000000000000000ULL;
const __uint128_t SMALL_RIGHT_EDGE = ((__uint128_t)0x100200400801002ULL << 64) | 0x0040080000000000ULL;
const __uint128_t SMALL_DOWN_EDGE = 0x7F80000000000ULL;
const __uint128_t SMALL_LEFT_EDGE = ((__uint128_t)0x8010020040080100ULL << 64) | 0x2004000000000000ULL;

const __uint128_t BOX_10 = ((__uint128_t)0xFFD00A0140280500ULL << 64) | 0xA01402805FF80000ULL;

const __uint128_t CENTER_21_BOX = ((__uint128_t)0xC000000ULL << 64) | 0x0000000000000000ULL;  // (3, 3), (4, 3)の2マス

void State_init(State* state);
bool eq_state(State* state1, State* state2);
Point_int color_p(State* state, int color);
void copy_state(State* new_state, State* old_state);
void set_row_wall_1(State* state, int x, int y);
void set_row_wall_0(State* state, int x, int y);
void set_column_wall_1(State* state, int x, int y);
void set_column_wall_0(State* state, int x, int y);

inline bool get_bit(__uint128_t bitarr, int x, int y);
void set_0(__uint128_t* bitarr, int x, int y);
void set_1(__uint128_t* bitarr, int x, int y);
bool* uint128ToBoolArray(uint64_t value_high, uint64_t value_low);

void print_bitarray(__uint128_t bitarr);
void print_state(State* state);
void print_full_bitarray(__uint128_t bitarr);
inline __uint128_t up_shift(__uint128_t bitarr);
inline __uint128_t right_shift(__uint128_t bitarr);
inline __uint128_t down_shift(__uint128_t bitarr);
inline __uint128_t left_shift(__uint128_t bitarr);
inline __uint128_t left_up_shift(__uint128_t bitarr);
inline __uint128_t right_down_shift(__uint128_t bitarr);
inline __uint128_t right_right_down_shift(__uint128_t bitarr);
inline __uint128_t right_down_down_shift(__uint128_t bitarr);
inline int min(int a, int b) {return (a < b) ? a : b;}
inline int max(int a, int b) {return (a > b) ? a : b;}

int select_action(float Q[ACTION_NUM], float N[ACTION_NUM], float P[ACTION_NUM],
float C_puct, float estimated_V, int color, int turn);
void movable_array(State* state, bool* mv, int x, int y, bool shortest_only);
bool accept_action_str(State* state, const char* s, bool check_placable, bool calc_placable_array, bool check_movable);
int get_player1_dist_from_goal(State* state);
int get_player2_dist_from_goal(State* state);
__uint128_t flip_bitarr(__uint128_t bitarr);
bool is_mirror_match(State* state);
bool placable_r_with_color(State* state, int x, int y, int color);
bool placable_c_with_color(State* state, int x, int y, int color);
__uint128_t calc_oneside_placable_r_cand_from_color(State* state, int color);
__uint128_t calc_oneside_placable_c_cand_from_color(State* state, int color);
bool is_certain_path_terminate(State* state, int color);
BitArrayPair placable_array(State* state, int color);

int arrivable_(State* state, int pawn_x, int pawn_y, int goal_y);
void calc_cross_bitarrs(State* state, __uint128_t row_bitarr, __uint128_t column_bitarr);
int arrivable_by_cross(__uint128_t cross_bitarrs[4], int pawn_x, int pawn_y, int goal_y);
void calc_dist_array(State* state, int goal_y);
void calc_dist_array_inner(uint8_t* dist_arr_p, int goal_y, __uint128_t cross_bitarrs[4]);
int arrivable_(State* state, int pawn_x, int pawn_y, int goal_y);
void calc_placable_array_and_set(State* state);
void calc_cross_bitarrs_global(__uint128_t row_bitarr, __uint128_t column_bitarr);
void calc_dist_array_inner(uint8_t* dist_arr_p, int goal_y, __uint128_t cross_bitarrs[4]);
BitArrayPair calc_placable_array_(State* state);

// ------Search_util.cpp------

typedef struct Tree {
    int N_arr[137];
    float W_arr[137];
    float Q_arr[137];
    struct Tree* children[137]; // 子ノードへのポインタ配列
} Tree;

Tree* createTree();
void deleteTree(Tree* tree);
void addChild(Tree* parent, int index, Tree* child);

}

#endif