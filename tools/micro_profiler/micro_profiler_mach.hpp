#pragma once
// micro_profiler_mach.hpp — macOS profiling using mach_absolute_time() with JSON reporting
// API:
//   ProfHandle h = prof_register("name");
//   prof_start(h); /* ... work ... */ prof_stop(h);
//   prof_report_json(stdout, /*pretty=*/true);

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <inttypes.h>
#include <mach/mach_time.h>

namespace microprof {

struct ProfHandle { uint32_t id; };

#ifndef MICROPROF_MAX_SLOTS
#define MICROPROF_MAX_SLOTS 65536u
#endif

// ---------- Mach timebase info (for converting mach_absolute_time to nanoseconds) ----------
static inline uint64_t mp_mach_timebase_ratio() {
    static uint64_t ratio = 0;
    if (ratio == 0) {
        mach_timebase_info_data_t info;
        if (mach_timebase_info(&info) == KERN_SUCCESS) {
            // ratio is numer/denom scaled by 2^32 to avoid floating point
            ratio = ((uint64_t)info.numer << 32) / info.denom;
        } else {
            ratio = 1ULL << 32; // fallback: 1:1
        }
    }
    return ratio;
}

static inline uint64_t mp_mach_to_ns(uint64_t mach_time) {
    // Convert mach_absolute_time ticks to nanoseconds
    uint64_t ratio = mp_mach_timebase_ratio();
    return (mach_time * ratio) >> 32;
}

// ---------- Registry (names/ids shared across threads) ----------
struct Registry {
    std::mutex mu;
    std::vector<std::string> names;
    std::atomic<uint32_t> next_id{0};

    static Registry& instance() { static Registry r; return r; }

    ProfHandle reg(const char* name) {
        std::lock_guard<std::mutex> lock(mu);
        if (names.size() >= MICROPROF_MAX_SLOTS) return ProfHandle{(uint32_t)(names.size()-1)};
        names.emplace_back(name ? name : "");
        return ProfHandle{next_id++};
    }
    const char* name(uint32_t id) {
        std::lock_guard<std::mutex> lock(mu);
        return id < names.size() ? names[id].c_str() : "<unknown>";
    }
    uint32_t count() const { return next_id.load(std::memory_order_relaxed); }
};

struct Slot {
    // Accumulators for this thread+handle
    uint64_t time_ticks = 0; // accumulated mach_absolute_time ticks
    uint64_t time_ns    = 0; // converted to nanoseconds

    // Start snapshots
    uint64_t start_ticks = 0;
    uint32_t nest        = 0;
};

struct ThreadData {
    std::vector<Slot> slots;

    ThreadData() { register_thread(this); }
    ~ThreadData(){ unregister_thread(this); }

    void ensure(uint32_t n) { if (slots.size() < n) slots.resize(n); }

    static std::vector<ThreadData*>& threads() { static auto* v = new std::vector<ThreadData*>(); return *v; }
    static std::mutex& thr_mu() { static auto* m = new std::mutex(); return *m; }
    static void register_thread(ThreadData* t){ std::lock_guard<std::mutex> lk(thr_mu()); threads().push_back(t); }
    static void unregister_thread(ThreadData* t){
        std::lock_guard<std::mutex> lk(thr_mu()); auto& v = threads();
        for (size_t i=0;i<v.size();++i){ if (v[i]==t){ v.erase(v.begin()+i); break; } }
    }
};

thread_local ThreadData mp_tls;

// ---------- Public API ----------
inline ProfHandle prof_register(const char* name) { return Registry::instance().reg(name); }

inline void prof_start(ProfHandle h) {
    auto& R = Registry::instance();
    uint32_t need = h.id + 1;
    if (mp_tls.slots.size() < need) { mp_tls.ensure(R.count()); if (mp_tls.slots.size() < need) mp_tls.ensure(need); }

    Slot& s = mp_tls.slots[h.id];
    if (s.nest++ == 0) {
        s.start_ticks = mach_absolute_time();
    }
}

inline void prof_stop(ProfHandle h) {
    uint64_t end_ticks = mach_absolute_time();
    if (h.id >= mp_tls.slots.size()) return;
    Slot& s = mp_tls.slots[h.id];
    if (s.nest == 0) return;
    if (--s.nest == 0) {
        uint64_t elapsed = end_ticks - s.start_ticks;
        s.time_ticks += elapsed;
        s.time_ns += mp_mach_to_ns(elapsed);
    }
}

inline void prof_reset() {
    std::lock_guard<std::mutex> lk(ThreadData::thr_mu());
    for (auto* t : ThreadData::threads()) {
        for (auto& s : t->slots) {
            s.time_ticks = s.time_ns = 0;
            s.start_ticks = 0;
            s.nest = 0;
        }
    }
}

// ---------- JSON reporting ----------
namespace detail {
    inline void json_escape(const char* s, std::string& out) {
        out.push_back('"');
        for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
            unsigned char c = *p;
            switch (c) {
                case '\"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (c < 0x20) {
                        char buf[7]; // \u00XX
                        snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else out.push_back((char)c);
            }
        }
        out.push_back('"');
    }

    struct Agg { uint64_t time_ticks=0, time_ns=0; };
}

// Report profiling results as JSON.
// Added parameter 'iterations': if >0, time is divided by this value
// to yield per-iteration averages. If iterations==0 we leave raw totals.
inline void prof_report_json(FILE* out = stdout, bool pretty = true, uint64_t iterations = 1) {
    Registry& R = Registry::instance();
    const uint32_t N = R.count();

    std::vector<detail::Agg> agg(N);
    {
        std::lock_guard<std::mutex> lk(ThreadData::thr_mu());
        for (auto* t : ThreadData::threads()) {
            const uint32_t upto = (uint32_t)std::min<size_t>(t->slots.size(), N);
            for (uint32_t i=0;i<upto;++i) {
                const Slot& s = t->slots[i];
                agg[i].time_ticks += s.time_ticks;
                agg[i].time_ns    += s.time_ns;
            }
        }
    }

    // Build a list of non-empty rows
    struct Row { uint32_t id; const char* name; uint64_t time_ticks; uint64_t time_ns; };
    std::vector<Row> rows;
    rows.reserve(N);
    for (uint32_t i=0;i<N;++i) {
        if (agg[i].time_ticks || agg[i].time_ns) {
            rows.push_back(Row{i, R.name(i), agg[i].time_ticks, agg[i].time_ns});
        }
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.time_ns > b.time_ns; });

    // Emit JSON
    std::string buf;
    buf.reserve(rows.size() * 160);
    auto nl = [&](bool cond){ if (pretty && cond) buf.push_back('\n'); };
    auto ind = [&](int n){ if (pretty) for (int i=0;i<n;++i) buf.push_back(' '); };

    buf.push_back('['); nl(!rows.empty());
    for (size_t idx=0; idx<rows.size(); ++idx) {
        const auto& r = rows[idx];
        const bool last = (idx+1==rows.size());
        const uint64_t adj_ticks = (iterations > 0 ? r.time_ticks / iterations : r.time_ticks);
        const uint64_t adj_ns    = (iterations > 0 ? r.time_ns / iterations : r.time_ns);
        ind(2); buf.push_back('{'); nl(true);
        ind(4); buf += "\"Function\": ";
        detail::json_escape(r.name ? r.name : "<unknown>", buf); buf.push_back(','); nl(true);
        ind(4); buf += "\"MACH_ABSOLUTE_TIME_TICKS\": "; { char tmp[32]; snprintf(tmp, sizeof(tmp), "%" PRIu64, (uint64_t)adj_ticks); buf += tmp; }
        buf.push_back(','); nl(true);
        ind(4); buf += "\"TIME_NS\": "; { char tmp[32]; snprintf(tmp, sizeof(tmp), "%" PRIu64, (uint64_t)adj_ns); buf += tmp; }
        if (iterations > 1) {
            buf.push_back(','); nl(true);
            ind(4); buf += "\"_iterations\": "; { char tmp[32]; snprintf(tmp, sizeof(tmp), "%" PRIu64, (uint64_t)iterations); buf += tmp; }
            nl(true);
        } else nl(true);
        nl(true);
        ind(2); buf.push_back('}'); if (!last) { buf.push_back(','); } nl(true);
    }
    buf.push_back(']'); nl(true);

    fwrite(buf.data(), 1, buf.size(), out);
    fflush(out);
}

// Optional: RAII helper & macro
struct ProfScope { ProfHandle h; explicit ProfScope(ProfHandle hh):h(hh){ prof_start(h); } ~ProfScope(){ prof_stop(h); } };
#define PROF_FN() static ::microprof::ProfHandle __mp_h = ::microprof::prof_register(__func__); \
                  ::microprof::ProfScope __mp_scope(__mp_h)

} // namespace microprof
