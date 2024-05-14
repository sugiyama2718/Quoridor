#include <cstdio>
#include <cstdint>
#include <cinttypes>
//#include <iostream>  //Could not find moduleが出る

extern "C" {

const int BOARD_LEN = 9;
const int BIT_BOARD_LEN = 11;
enum DIRECTION {
    UP,
    RIGHT,
    DOWN,
    LEFT
};
const __uint128_t BIT_BOARD_MASK = ((__uint128_t)0xFF9FF3FE7FCFF9FFULL << 64) | 0x3FE7FCFF80000000ULL;
const __uint128_t UP_EDGE = ((__uint128_t)0xFF80000000000000ULL << 64) | 0x0000000000000000ULL;
const __uint128_t RIGHT_EDGE = ((__uint128_t)0x80100200400801ULL << 64) | 0x0020040080000000ULL;
const __uint128_t DOWN_EDGE = 0xFF80000000ULL;
const __uint128_t LEFT_EDGE = ((__uint128_t)0x8010020040080100ULL << 64) | 0x2004008000000000ULL;
const __uint128_t BOX_10 = ((__uint128_t)0XFFD00A0140280500ULL << 64) | 0xA01402805FF80000ULL;
__uint128_t cross_bitarrs[4];

void print_bitarray(__uint128_t bitarr);
void print_full_bitarray(__uint128_t bitarr);
inline __uint128_t up_shift(__uint128_t bitarr);
inline __uint128_t right_shift(__uint128_t bitarr);
inline __uint128_t down_shift(__uint128_t bitarr);
inline __uint128_t left_shift(__uint128_t bitarr);
inline __uint128_t right_down_shift(__uint128_t bitarr);
int arrivable_(uint64_t row_bitarr_high, uint64_t row_bitarr_low, uint64_t column_bitarr_high, uint64_t column_bitarr_low, int pawn_x, int pawn_y, int goal_y);

struct BitArrayPair {
    __uint128_t bitarr1;
    __uint128_t bitarr2;
};

struct Point {
    uint8_t x, y;
};

inline bool get_bit(__uint128_t bitarr, int x, int y) {
    return (bitarr & ((__uint128_t)1 << (127 - (x + y * BIT_BOARD_LEN)))) > 0;
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

void calc_cross_bitarrs(__uint128_t row_bitarr, __uint128_t column_bitarr) {
    cross_bitarrs[UP] = UP_EDGE;
    cross_bitarrs[UP] |= down_shift(row_bitarr);
    cross_bitarrs[UP] |= right_down_shift(row_bitarr);

    cross_bitarrs[RIGHT] = RIGHT_EDGE;
    cross_bitarrs[RIGHT] |= column_bitarr;
    cross_bitarrs[RIGHT] |= down_shift(column_bitarr);

    cross_bitarrs[DOWN] = DOWN_EDGE;
    cross_bitarrs[DOWN] |= row_bitarr;
    cross_bitarrs[DOWN] |= right_shift(row_bitarr);

    cross_bitarrs[LEFT] = LEFT_EDGE;
    cross_bitarrs[LEFT] |= right_shift(column_bitarr);
    cross_bitarrs[LEFT] |= right_down_shift(column_bitarr);

    for(int i = 0; i < 4; i++) {
        cross_bitarrs[i] = ~cross_bitarrs[i];
        cross_bitarrs[i] &= BIT_BOARD_MASK;
    }
}

int arrivable_by_uint128(__uint128_t row_bitarr, __uint128_t column_bitarr, int pawn_x, int pawn_y, int goal_y) {
    calc_cross_bitarrs(row_bitarr, column_bitarr);

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

uint8_t* calc_dist_array(uint64_t row_bitarr_high, uint64_t row_bitarr_low, uint64_t column_bitarr_high, uint64_t column_bitarr_low, int goal_y) {
    __uint128_t row_bitarr = ((__uint128_t)row_bitarr_high << 64) | row_bitarr_low;
    __uint128_t column_bitarr = ((__uint128_t)column_bitarr_high << 64) | column_bitarr_low;

    calc_cross_bitarrs(row_bitarr, column_bitarr);

    Point point_queue[BOARD_LEN * BOARD_LEN];
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
            get_bit(cross_bitarrs[i], x2, y2) && (dist_array_ret[x3 + y3 * BOARD_LEN] > dist_array_ret[x2 + y2 * BOARD_LEN] + 1)) {
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

int arrivable_(uint64_t row_bitarr_high, uint64_t row_bitarr_low, uint64_t column_bitarr_high, uint64_t column_bitarr_low, int pawn_x, int pawn_y, int goal_y) {
    __uint128_t row_bitarr = ((__uint128_t)row_bitarr_high << 64) | row_bitarr_low;
    __uint128_t column_bitarr = ((__uint128_t)column_bitarr_high << 64) | column_bitarr_low;
    
    return arrivable_by_uint128(row_bitarr, column_bitarr, pawn_x, pawn_y, goal_y);
}

BitArrayPair calc_placable_array_(uint64_t row_bitarr_high, uint64_t row_bitarr_low, uint64_t column_bitarr_high, uint64_t column_bitarr_low, 
int pawn_1p_x, int pawn_1p_y, int pawn_2p_x, int pawn_2p_y) {
    __uint128_t row_bitarr = ((__uint128_t)row_bitarr_high << 64) | row_bitarr_low;
    __uint128_t column_bitarr = ((__uint128_t)column_bitarr_high << 64) | column_bitarr_low;

    // 各交点について壁に触れているなら1が立っているbitarrayを作る
    __uint128_t wall_point_bitarr = BOX_10;
    wall_point_bitarr |= down_shift(row_bitarr);
    wall_point_bitarr |= right_down_shift(row_bitarr);
    wall_point_bitarr |= right_right_down_shift(row_bitarr);
    wall_point_bitarr |= right_shift(column_bitarr);
    wall_point_bitarr |= right_down_shift(column_bitarr);
    wall_point_bitarr |= right_down_down_shift(column_bitarr);

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
    ret.bitarr1 = ~(row_bitarr | column_bitarr | left_shift(row_bitarr) | right_shift(row_bitarr));
    ret.bitarr2 = ~(row_bitarr | column_bitarr | up_shift(column_bitarr) | down_shift(column_bitarr));

    // 既においてある壁とぶつかるのを除外
    must_be_checked_x &= ret.bitarr2;
    must_be_checked_y &= ret.bitarr1;

    __uint128_t virtual_row_wall, virtual_column_wall;
    for(int i = 0;i < 128;i++) {
        virtual_row_wall = ((__uint128_t)1 << i) & must_be_checked_y;
        if(virtual_row_wall > 0) {
            if(!arrivable_by_uint128(row_bitarr | virtual_row_wall, column_bitarr, pawn_1p_x, pawn_1p_y, 0)
            || !arrivable_by_uint128(row_bitarr | virtual_row_wall, column_bitarr, pawn_2p_x, pawn_2p_y, BOARD_LEN - 1)) ret.bitarr1 &= ~((__uint128_t)1 << i);
        }

        virtual_column_wall = ((__uint128_t)1 << i) & must_be_checked_x;
        if(virtual_column_wall > 0) {
            if(!arrivable_by_uint128(row_bitarr, column_bitarr | virtual_column_wall, pawn_1p_x, pawn_1p_y, 0)
            || !arrivable_by_uint128(row_bitarr, column_bitarr | virtual_column_wall, pawn_2p_x, pawn_2p_y, BOARD_LEN - 1)) ret.bitarr2 &= ~((__uint128_t)1 << i);
        }
    }

    return ret;
}

}
