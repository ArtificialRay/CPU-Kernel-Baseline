#include "loop_002.h"
#include <limits>

extern "C" void inner_loop_002(struct loop_002_data *data);

extern "C" int armbench_entry_loop_002(void *a, void *b, int64_t n, void *res_out) {
    if (n < 0 || n > std::numeric_limits<int>::max()) {
        return -1;
    }
    struct loop_002_data data;
    data.a = static_cast<uint32_t *>(a);
    data.b = static_cast<uint32_t *>(b);
    data.n = static_cast<int>(n);
    data.res = 0u;
    inner_loop_002(&data);
    *static_cast<uint32_t *>(res_out) = data.res;
    return 0;
}
