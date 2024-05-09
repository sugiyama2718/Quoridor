#include <cstdio>
#include <cstdint>
#include <cinttypes>
//#include <iostream>  //Could not find moduleが出る

const int BOARD_LEN = 9;
const int BIT_BOARD_LEN = 11;
const __uint128_t BIT_BOARD_MASK = ((__uint128_t)0xFF9FF3FE7FCFF9FFULL << 64) | 0x3FE7FCFF80000000ULL;
const __uint128_t UP_EDGE = ((__uint128_t)0xFF80000000000000ULL << 64) | 0x0000000000000000ULL;
const __uint128_t RIGHT_EDGE = ((__uint128_t)0x80100200400801ULL << 64) | 0x0020040080000000ULL;
const __uint128_t DOWN_EDGE = 0xFF80000000ULL;
const __uint128_t LEFT_EDGE = ((__uint128_t)0x8010020040080100ULL << 64) | 0x2004008000000000ULL;

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
    print_full_bitarray(UP_EDGE);
    // printf("\n");
    // print_full_bitarray(RIGHT_EDGE);

    return (row_bitarr < 100);
}
}
