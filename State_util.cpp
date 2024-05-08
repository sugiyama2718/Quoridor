extern "C" {
    int arrivable_(long long row_bitarr_high, long long row_bitarr_low) {
        __uint128_t row_bitarr = ((__uint128_t)row_bitarr_high << 64) | row_bitarr_low;
        return (row_bitarr < 100);
    }
}
