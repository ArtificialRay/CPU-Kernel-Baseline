// bench/datasets/_ncnn_lib/_mat_factory.cpp
//
// Python ↔ ncnn::Mat bridge. compile.py always concatenates this source into
// the solution.so build, so Python (via ctypes) and the armbench_entry_*
// shim (C++) operate on the same ncnn::Mat ABI — no cross-.so vtable risk.
//
// Layout convention: 3D Mat is (c, h, w). 1D Mat is (w,). All float32.
// Data is copied IN from numpy at create time and copied OUT at read time;
// ctypes never has to know about ncnn::Mat's internal cstep/padding layout.

#include "mat.h"
#include "option.h"
#include <cstring>
#include <cstdlib>

extern "C" {

// ── Mat construction ─────────────────────────────────────────────────────

// Create a 3D Mat (c, h, w) and copy `data` (c*h*w float32, C-major) into it.
// Returns an opaque pointer that Python keeps until armbench_ncnn_mat_destroy.
void* armbench_ncnn_mat_create_3d(int w, int h, int c, const float* data)
{
    auto* m = new ncnn::Mat();
    m->create(w, h, c, (size_t)4u, (ncnn::Allocator*)0);
    if (m->empty()) { delete m; return nullptr; }
    // Honor ncnn's cstep padding by writing through channel(cc).row(hh).
    for (int cc = 0; cc < c; ++cc) {
        for (int hh = 0; hh < h; ++hh) {
            float* dst = m->channel(cc).row(hh);
            const float* src = data + cc * h * w + hh * w;
            std::memcpy(dst, src, w * sizeof(float));
        }
    }
    return m;
}

// Create a 2D Mat (h, w) — used for Conv1D where shape is (channels=h, seq_len=w).
void* armbench_ncnn_mat_create_2d(int w, int h, const float* data)
{
    auto* m = new ncnn::Mat();
    m->create(w, h, (size_t)4u, (ncnn::Allocator*)0);
    if (m->empty()) { delete m; return nullptr; }
    for (int hh = 0; hh < h; ++hh) {
        std::memcpy(m->row(hh), data + hh * w, w * sizeof(float));
    }
    return m;
}

// Create a 1D Mat (w,) — used for weights (flat, out_c*in_c*kh*kw) and bias.
void* armbench_ncnn_mat_create_1d(int w, const float* data)
{
    auto* m = new ncnn::Mat();
    m->create(w, (size_t)4u, (ncnn::Allocator*)0);
    if (m->empty()) { delete m; return nullptr; }
    std::memcpy((float*)m->data, data, w * sizeof(float));
    return m;
}

// Create an empty Mat (no allocation). Used for output blob — the harness's
// armbench_entry_conv2d calls .create() on it once output dims are known.
void* armbench_ncnn_mat_create_empty()
{
    return new ncnn::Mat();
}

void armbench_ncnn_mat_destroy(void* m_v)
{
    delete reinterpret_cast<ncnn::Mat*>(m_v);
}

// ── Mat introspection (for output readback) ──────────────────────────────

int armbench_ncnn_mat_dims(void* m_v)
{ return reinterpret_cast<ncnn::Mat*>(m_v)->dims; }

int armbench_ncnn_mat_w(void* m_v)
{ return reinterpret_cast<ncnn::Mat*>(m_v)->w; }

int armbench_ncnn_mat_h(void* m_v)
{ return reinterpret_cast<ncnn::Mat*>(m_v)->h; }

int armbench_ncnn_mat_c(void* m_v)
{ return reinterpret_cast<ncnn::Mat*>(m_v)->c; }

int armbench_ncnn_mat_empty(void* m_v)
{ return reinterpret_cast<ncnn::Mat*>(m_v)->empty() ? 1 : 0; }

// Read a 3D Mat (c, h, w) into a flat C-major float32 buffer (caller-allocated,
// size c*h*w). Returns 0 on success, -1 on shape mismatch.
int armbench_ncnn_mat_read_3d(void* m_v, float* out)
{
    const auto& m = *reinterpret_cast<ncnn::Mat*>(m_v);
    if (m.empty() || m.dims != 3) return -1;
    for (int cc = 0; cc < m.c; ++cc) {
        for (int hh = 0; hh < m.h; ++hh) {
            const float* src = m.channel(cc).row(hh);
            std::memcpy(out + cc * m.h * m.w + hh * m.w, src, m.w * sizeof(float));
        }
    }
    return 0;
}

int armbench_ncnn_mat_read_2d(void* m_v, float* out)
{
    const auto& m = *reinterpret_cast<ncnn::Mat*>(m_v);
    if (m.empty() || m.dims != 2) return -1;
    for (int hh = 0; hh < m.h; ++hh) {
        std::memcpy(out + hh * m.w, m.row(hh), m.w * sizeof(float));
    }
    return 0;
}

int armbench_ncnn_mat_read_1d(void* m_v, float* out)
{
    const auto& m = *reinterpret_cast<ncnn::Mat*>(m_v);
    if (m.empty() || m.dims != 1) return -1;
    std::memcpy(out, (const float*)m.data, m.w * sizeof(float));
    return 0;
}

// ── Option construction ──────────────────────────────────────────────────

// Default Option matching today's make_opt() in starter/ncnn/ncnn_helpers.h:
//   num_threads=1, no packing, no fp16/bf16, no sgemm/winograd shortcuts.
void* armbench_ncnn_option_create_default()
{
    auto* opt = new ncnn::Option();
    opt->num_threads = 1;
    opt->use_packing_layout = false;
    opt->use_fp16_storage = false;
    opt->use_bf16_storage = false;
    opt->use_sgemm_convolution = false;
    opt->use_winograd_convolution = false;
    return opt;
}

void armbench_ncnn_option_destroy(void* opt_v)
{
    delete reinterpret_cast<ncnn::Option*>(opt_v);
}

} // extern "C"
