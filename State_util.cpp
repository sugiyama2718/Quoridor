#include <cstdio>
#include <cstdint>
#include <cinttypes>
//#include <iostream>  //Could not find moduleが出る

extern "C" {
    int arrivable_(uint64_t row_bitarr_high, uint64_t row_bitarr_low, uint64_t column_bitarr_high, uint64_t column_bitarr_low, int pawn_x, int pawn_y, int goal_y) {
        __uint128_t row_bitarr = ((__uint128_t)row_bitarr_high << 64) | row_bitarr_low;
        __uint128_t column_bitarr = ((__uint128_t)column_bitarr_high << 64) | column_bitarr_low;
        printf("%" PRIu64 " %" PRIu64 "\n", row_bitarr_high, row_bitarr_low);
        for(int y = 0; y < 8; y++) {
            for(int x = 0; x < 8; x++) {
                __uint128_t shifted = (row_bitarr << (x + y * 11)) >> 64;
                printf("%d", (uint64_t)(shifted & 0x8000000000000000) >> 63);
            }
            printf("\n");
        }
        return (row_bitarr < 100);
    }
}
