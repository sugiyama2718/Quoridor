#include <cstdio>
extern "C" {
    int arrivable_(long long row_bitarr_high, long long row_bitarr_low, long long column_bitarr_high, long long column_bitarr_low, int x, int y, int goal_y) {
        __uint128_t row_bitarr = ((__uint128_t)row_bitarr_high << 64) | row_bitarr_low;
        __uint128_t column_bitarr = ((__uint128_t)column_bitarr_high << 64) | column_bitarr_low;
        printf("%d %d %d\n", x, y, goal_y);
        return (row_bitarr < 100);
    }
}
