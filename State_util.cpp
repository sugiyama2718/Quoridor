#include <cstdio>
#include <cstdint>
#include <cinttypes>
//#include <iostream>  //Could not find moduleが出る

const int BOARD_LEN = 9;
const int BIT_BOARD_LEN = 11;

void print_bitarray(__uint128_t bitarr) {
    for(int y = 0; y < BOARD_LEN - 1; y++) {
        for(int x = 0; x < BOARD_LEN - 1; x++) {
            __uint128_t shifted = (bitarr << (x + y * BIT_BOARD_LEN)) >> 64;
            printf("%d", (uint64_t)(shifted & 0x8000000000000000) >> 63);
        }
        printf("\n");
    }
}

inline __uint128_t up_shift(__uint128_t bitarr){
    return bitarr << BIT_BOARD_LEN;
}

inline __uint128_t right_shift(__uint128_t bitarr){
    return bitarr >> 1;
}

inline __uint128_t down_shift(__uint128_t bitarr){
    return bitarr >> BIT_BOARD_LEN;
}

inline __uint128_t left_shift(__uint128_t bitarr){
    return bitarr << 1;
}

inline __uint128_t right_down_shift(__uint128_t bitarr){
    return bitarr >> (BIT_BOARD_LEN + 1);
}

extern "C" {
int arrivable_(uint64_t row_bitarr_high, uint64_t row_bitarr_low, uint64_t column_bitarr_high, uint64_t column_bitarr_low, int pawn_x, int pawn_y, int goal_y) {
    __uint128_t row_bitarr = ((__uint128_t)row_bitarr_high << 64) | row_bitarr_low;
    __uint128_t column_bitarr = ((__uint128_t)column_bitarr_high << 64) | column_bitarr_low;
    
    printf("----------\n");
    print_bitarray(row_bitarr);
    printf("\n");
    print_bitarray(right_down_shift(row_bitarr));

    return (row_bitarr < 100);
}
}
