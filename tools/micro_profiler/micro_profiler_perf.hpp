#pragma once
// micro_profiler_perf.hpp — per-thread cycles & instructions with JSON reporting
// API:
//   ProfHandle h = prof_register("name");
//   prof_start(h); /* ... work ... */ prof_stop(h);
//   prof_report_json(stdout, /*pretty=*/true);

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <errno.h>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <inttypes.h>

namespace microprof {

struct ProfHandle { uint32_t id; };

#ifndef MICROPROF_MAX_SLOTS
#define MICROPROF_MAX_SLOTS 65536u
#endif

#ifndef MICROPROF_EXCLUDE_KERNEL
#define MICROPROF_EXCLUDE_KERNEL 1
#endif
#ifndef MICROPROF_EXCLUDE_HV
#define MICROPROF_EXCLUDE_HV 1
#endif

// ---------- Optional cycles-only fallback via CNTVCT_EL0 ----------
#if defined(__aarch64__) && defined(MICROPROF_FALLBACK_CNTVCT)
static inline uint64_t mp_read_cntvct() { uint64_t v; asm volatile("mrs %0, cntvct_el0" : "=r"(v)); return v; }
static inline uint64_t mp_cntfrq() { uint64_t f; asm volatile("mrs %0, cntfrq_el0" : "=r"(f)); return f ? f : 1; }
#endif

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

// ---------- Per-thread perf group ----------
struct PerfGroup {
    int fd_leader = -1;      // cycles
    int fd_insts  = -1;      // instructions
    int fd_time   = -1;      // task clock (ns of on-CPU time)
    bool ok       = false;
    bool has_insts= false;
    bool has_time = false;
#if defined(__aarch64__) && defined(MICROPROF_FALLBACK_CNTVCT)
    bool fallback = false;
    uint64_t cntfrq = 0;
#endif

    // PERF_FORMAT_GROUP layout: [0]=nr, then nr u64 values (cycles, insts?, task_clock?)
    struct ReadBuf { uint64_t nr; uint64_t val[3]; };

    static long sys_perf_event_open(struct perf_event_attr* attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
#ifdef __NR_perf_event_open
        return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
#else
        errno = ENOSYS; return -1;
#endif
    }

    void init() {
        if (ok
#if defined(__aarch64__) && defined(MICROPROF_FALLBACK_CNTVCT)
            || fallback
#endif
            ) return;

        // Leader: CPU cycles
        perf_event_attr pe{};
        pe.size = sizeof(pe);
        pe.type = PERF_TYPE_HARDWARE;
        pe.config = PERF_COUNT_HW_CPU_CYCLES;
        pe.read_format = PERF_FORMAT_GROUP;
        pe.disabled = 0; // always on; we subtract start/stop snapshots
        pe.exclude_kernel = MICROPROF_EXCLUDE_KERNEL ? 1u : 0u;
        pe.exclude_hv     = MICROPROF_EXCLUDE_HV ? 1u : 0u;
        pe.exclude_idle   = 0;

        fd_leader = (int)sys_perf_event_open(&pe, /*pid=*/0, /*cpu=*/-1, /*group_fd=*/-1, /*flags=*/0);
        if (fd_leader < 0) {
#if defined(__aarch64__) && defined(MICROPROF_FALLBACK_CNTVCT)
            fallback = true; cntfrq = mp_cntfrq();
#endif
            return;
        }

        // Child: instructions retired
        perf_event_attr pi{};
        pi.size = sizeof(pi);
        pi.type = PERF_TYPE_HARDWARE;
        pi.config = PERF_COUNT_HW_INSTRUCTIONS;
        pi.disabled = 0;
        pi.exclude_kernel = MICROPROF_EXCLUDE_KERNEL ? 1u : 0u;
        pi.exclude_hv     = MICROPROF_EXCLUDE_HV ? 1u : 0u;

        fd_insts = (int)sys_perf_event_open(&pi, 0, -1, fd_leader, 0);
        if (fd_insts >= 0) has_insts = true;

        // Child: per-thread task clock (nanoseconds executing on CPU)
        perf_event_attr pt{};
        pt.size = sizeof(pt);
        pt.type = PERF_TYPE_SOFTWARE;
        pt.config = PERF_COUNT_SW_TASK_CLOCK; // ns of scheduled CPU time
        pt.disabled = 0;
        fd_time = (int)sys_perf_event_open(&pt, 0, -1, fd_leader, 0);
        if (fd_time >= 0) has_time = true;

        ok = (fd_leader >= 0);
    }

    inline bool is_ready() const {
#if defined(__aarch64__) && defined(MICROPROF_FALLBACK_CNTVCT)
        return ok || fallback;
#else
        return ok;
#endif
    }

    inline void read(uint64_t& cycles, uint64_t& insts, bool& insts_valid, uint64_t& task_ns, bool& time_valid) {
        cycles = 0; insts = 0; insts_valid = false; task_ns = 0; time_valid = false;
#if defined(__aarch64__) && defined(MICROPROF_FALLBACK_CNTVCT)
        if (!ok && fallback) { cycles = mp_read_cntvct(); insts_valid = false; return; }
#endif
        if (!ok) return;
        ReadBuf buf{};
        int num_events = 1 + (has_insts ? 1 : 0) + (has_time ? 1 : 0);
        const ssize_t need = (ssize_t)sizeof(uint64_t) * (1 + num_events);
        const ssize_t got  = ::read(fd_leader, &buf, need);
        if (got == need && buf.nr >= 1) {
            int idx = 0;
            cycles = buf.val[idx++];
            if (has_insts && buf.nr > idx) { insts = buf.val[idx++]; insts_valid = true; }
            if (has_time && buf.nr > idx) { task_ns = buf.val[idx++]; time_valid = true; }
        }
    }

    ~PerfGroup() {
        if (fd_insts >= 0) ::close(fd_insts);
        if (fd_time  >= 0) ::close(fd_time);
        if (fd_leader >= 0) ::close(fd_leader);
    }
};

struct Slot {
    // Accumulators for this thread+handle
    uint64_t cycles = 0;
    uint64_t insts  = 0;
    uint64_t time_ns = 0; // accumulated task clock ns

    // Start snapshots
    uint64_t start_cycles = 0;
    uint64_t start_insts  = 0;
    uint64_t start_time_ns= 0;
    uint32_t nest         = 0;
    bool     insts_valid  = false; // true if instructions available for this thread
    bool     time_valid   = false; // true if task clock available
};

struct ThreadData {
    PerfGroup pg;
    std::vector<Slot> slots;

    ThreadData() { pg.init(); register_thread(this); }
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

    uint64_t c=0, i=0, t=0; bool iv=false, tv=false;
    mp_tls.pg.read(c, i, iv, t, tv);

    Slot& s = mp_tls.slots[h.id];
    if (s.nest++ == 0) {
        s.start_cycles = c;
        s.start_insts  = i;
        s.insts_valid  = iv;
        s.start_time_ns= t;
        s.time_valid   = tv;
    }
}

inline void prof_stop(ProfHandle h) {
    if (h.id >= mp_tls.slots.size()) return;
    Slot& s = mp_tls.slots[h.id];
    if (s.nest == 0) return;
    if (--s.nest == 0) {
        uint64_t c=0, i=0, t=0; bool iv=false, tv=false;
        mp_tls.pg.read(c, i, iv, t, tv);
        s.cycles += (c - s.start_cycles);
        if (s.insts_valid && iv) s.insts += (i - s.start_insts);
        if (s.time_valid && tv) s.time_ns += (t - s.start_time_ns);
        // if instructions unavailable, we keep insts at 0 and CPI will be null
    }
}

inline void prof_reset() {
    std::lock_guard<std::mutex> lk(ThreadData::thr_mu());
    for (auto* t : ThreadData::threads()) {
        for (auto& s : t->slots) {
            s.cycles = s.insts = s.time_ns = 0;
            s.start_cycles = s.start_insts = s.start_time_ns = 0;
            s.nest = 0; s.insts_valid = false; s.time_valid = false;
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

    struct Agg { uint64_t cycles=0, insts=0, time_ns=0; bool any_insts=false; bool any_time=false; };
}

// Report profiling results as JSON.
// Added parameter 'iterations': if >0, cycles & instructions are divided by this value
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
                agg[i].cycles += s.cycles;
                agg[i].insts  += s.insts;
                agg[i].time_ns+= s.time_ns;
                if (s.insts_valid) agg[i].any_insts = true;
                if (s.time_valid)  agg[i].any_time  = true;
            }
        }
    }

    // Build a list of non-empty rows
    struct Row { uint32_t id; const char* name; uint64_t cycles; uint64_t insts; uint64_t time_ns; bool any_insts; bool any_time; };
    std::vector<Row> rows;
    rows.reserve(N);
    for (uint32_t i=0;i<N;++i) {
        if (agg[i].cycles || agg[i].insts || agg[i].time_ns) {
            rows.push_back(Row{i, R.name(i), agg[i].cycles, agg[i].insts, agg[i].time_ns, agg[i].any_insts, agg[i].any_time});
        }
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.cycles > b.cycles; });

    // Emit JSON
    std::string buf;
    buf.reserve(rows.size() * 160);
    auto nl = [&](bool cond){ if (pretty && cond) buf.push_back('\n'); };
    auto ind = [&](int n){ if (pretty) for (int i=0;i<n;++i) buf.push_back(' '); };

    buf.push_back('['); nl(!rows.empty());
    for (size_t idx=0; idx<rows.size(); ++idx) {
        const auto& r = rows[idx];
        const bool last = (idx+1==rows.size());
        const uint64_t adj_cycles = (iterations > 0 ? r.cycles / iterations : r.cycles);
        const uint64_t adj_insts  = (iterations > 0 ? r.insts  / iterations : r.insts);
        const uint64_t adj_time_ns= (iterations > 0 ? r.time_ns/ iterations : r.time_ns);
        ind(2); buf.push_back('{'); nl(true);
        ind(4); buf += "\"Function\": ";
        detail::json_escape(r.name ? r.name : "<unknown>", buf); buf.push_back(','); nl(true);
        ind(4); buf += "\"PERF_COUNT_HW_CPU_CYCLES\": "; { char tmp[32]; snprintf(tmp, sizeof(tmp), "%" PRIu64, (uint64_t)adj_cycles); buf += tmp; }
        buf.push_back(','); nl(true);
        ind(4); buf += "\"PERF_COUNT_HW_INSTRUCTIONS\": ";
        if (r.any_insts) { char tmp[32]; snprintf(tmp, sizeof(tmp), "%" PRIu64, (uint64_t)adj_insts); buf += tmp; }
        else { if (adj_insts == 0) buf += "null"; else { char tmp[32]; snprintf(tmp, sizeof(tmp), "%" PRIu64, (uint64_t)adj_insts); buf += tmp; } }
        buf.push_back(','); nl(true);
        ind(4); buf += "\"PERF_COUNT_SW_TASK_CLOCK_NS\": ";
        if (r.any_time) { char tmp[32]; snprintf(tmp, sizeof(tmp), "%" PRIu64, (uint64_t)adj_time_ns); buf += tmp; }
        else { if (adj_time_ns == 0) buf += "null"; else { char tmp[32]; snprintf(tmp, sizeof(tmp), "%" PRIu64, (uint64_t)adj_time_ns); buf += tmp; } }
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

// Optional: RAII helper & macro remain available if you want them later.
// They’re no-ops for call counting (which we removed).
struct ProfScope { ProfHandle h; explicit ProfScope(ProfHandle hh):h(hh){ prof_start(h); } ~ProfScope(){ prof_stop(h); } };
#define PROF_FN() static ::microprof::ProfHandle __mp_h = ::microprof::prof_register(__func__); \
                  ::microprof::ProfScope __mp_scope(__mp_h)

} // namespace microprof
