#include "loop_003.h"
#include <cassert>

extern "C" void inner_loop_003(struct loop_003_data *data);

extern "C" int armbench_entry_loop_003(void *a, void *b, int64_t n, void *res_out) {
    struct loop_003_data data;
    data.a = static_cast<double *>(a);
    data.b = static_cast<double *>(b);
    assert(n <= INT_MAX && "N exceeds int range");
    data.n = static_cast<int>(n);
    data.res = 0.0;
    inner_loop_003(&data);
    *static_cast<double *>(res_out) = data.res;
    return 0;
}
