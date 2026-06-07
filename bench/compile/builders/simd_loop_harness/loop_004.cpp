#include "loop_004.h"
#include <limits>

extern "C" void inner_loop_004(struct loop_004_data *data);

extern "C" int armbench_entry_loop_004(void *a, void *b, int64_t n, void *res_out) {
    if (n < 0 || n > std::numeric_limits<int>::max()) {
        return -1;
    }
    struct loop_004_data data;
    data.a = static_cast<uint64_t *>(a);
    data.b = static_cast<uint64_t *>(b);
    data.n = static_cast<int>(n);
    data.res = 0ull;
    inner_loop_004(&data);
    *static_cast<uint64_t *>(res_out) = data.res;
    return 0;
}
