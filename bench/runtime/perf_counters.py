"""In-process hardware perf counters via the Linux ``perf_event_open(2)`` syscall.

This is the cpu-side counterpart of ``perf stat`` for an in-process ctypes
kernel call. We open a counter *group* — leader ``CPU_CYCLES`` plus
``INSTRUCTIONS`` and ``CACHE_MISSES`` — then RESET/ENABLE around the timed loop
in `bench/runtime/timing.py` and read the aggregate once. The group read keeps
all three counters covering exactly the same execution window, so
``ipc = instructions / cycles`` is meaningful.

Cycles is the canonical metric for arm-bench because it is frequency-independent
(comparable across DVFS / cores of the same ISA), unlike wall-clock ns.

Graceful degradation
--------------------
Everything here is best-effort. If the syscall is unavailable (non-Linux, no
``perf_event_open``) or denied (``/proc/sys/kernel/perf_event_paranoid`` too
strict → ``EACCES``), `open_counters()` returns ``None`` and the caller falls
back to ns-only timing — no exception escapes.

Caveat
------
The ENABLE/DISABLE window wraps the whole Python ``repeat`` loop in
`time_callable`, so counts include the loop overhead + ctypes marshalling around
``entry(*ctx.entry_args)``, not just the kernel. Negligible at ms-scale convs
where the kernel dominates; bump ``inner_iters`` to amortize it for fast kernels.
"""

from __future__ import annotations

import ctypes
import fcntl
import logging
import os
import platform
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── perf_event ABI constants (from linux/perf_event.h) ────────────────────────

PERF_TYPE_HARDWARE = 0
PERF_COUNT_HW_CPU_CYCLES = 0
PERF_COUNT_HW_INSTRUCTIONS = 1
PERF_COUNT_HW_CACHE_MISSES = 3

PERF_FORMAT_TOTAL_TIME_ENABLED = 1 << 0
PERF_FORMAT_TOTAL_TIME_RUNNING = 1 << 1
PERF_FORMAT_GROUP = 1 << 3

# _IO('$', n) → ('$' << 8) | n, with '$' == 0x24. No size field on these ioctls.
PERF_EVENT_IOC_ENABLE = 0x2400
PERF_EVENT_IOC_DISABLE = 0x2401
PERF_EVENT_IOC_RESET = 0x2403
PERF_IOC_FLAG_GROUP = 1

_ATTR_DISABLED = 1 << 0  # flags bit 0: start the leader disabled

# __NR_perf_event_open is arch-specific.
_SYSCALL_NR = {
    "aarch64": 241,
    "x86_64": 298,
}


class perf_event_attr(ctypes.Structure):
    """perf_event_attr truncated to PERF_ATTR_SIZE_VER1 (72 bytes).

    We only set type/size/config/read_format/flags; the union members
    (sample_period, wakeup_events, config1/2) stay zero. The kernel accepts this
    size and zero-extends the rest.
    """

    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),  # union sample_period / sample_freq
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),          # bitfield: bit0 = disabled, ...
        ("wakeup_events", ctypes.c_uint32),  # union wakeup_events / watermark
        ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64),        # union bp_addr / config1
        ("config2", ctypes.c_uint64),        # union bp_len / config2
    ]


# Order is load-bearing: a group read returns values in counter-creation order,
# which we map back to these names.
_EVENTS = (
    ("cycles", PERF_COUNT_HW_CPU_CYCLES),
    ("instructions", PERF_COUNT_HW_INSTRUCTIONS),
    ("cache_misses", PERF_COUNT_HW_CACHE_MISSES),
)


@dataclass
class CounterTotals:
    """Aggregate counts over the whole measured window (not yet per-iter)."""

    cycles: Optional[int]
    instructions: Optional[int]
    cache_misses: Optional[int]


def _libc() -> Optional[ctypes.CDLL]:
    try:
        return ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return None


class PerfCounters:
    """An open perf_event counter group. Use via `open_counters()`.

    Lifecycle: `reset()` → `enable()` → (run kernel loop) → `disable()` →
    `read()` → `close()`. All ioctls target the whole group via the leader fd.
    """

    def __init__(self, libc: ctypes.CDLL, syscall_nr: int) -> None:
        self._libc = libc
        self._nr = syscall_nr
        self._fds: list[int] = []
        self._leader = -1

    # ── open / close ──────────────────────────────────────────────────────────

    def _perf_event_open(self, attr: perf_event_attr, group_fd: int) -> int:
        # syscall(nr, attr*, pid=0 (this process), cpu=-1 (any), group_fd, flags=0)
        return self._libc.syscall(
            ctypes.c_long(self._nr),
            ctypes.byref(attr),
            ctypes.c_int(0),
            ctypes.c_int(-1),
            ctypes.c_int(group_fd),
            ctypes.c_ulong(0),
        )

    def _open_group(self) -> bool:
        read_format = (
            PERF_FORMAT_GROUP
            | PERF_FORMAT_TOTAL_TIME_ENABLED
            | PERF_FORMAT_TOTAL_TIME_RUNNING
        )
        for i, (_, config) in enumerate(_EVENTS):
            attr = perf_event_attr()
            attr.size = ctypes.sizeof(perf_event_attr)
            attr.type = PERF_TYPE_HARDWARE
            attr.config = config
            attr.read_format = read_format
            if i == 0:
                attr.flags = _ATTR_DISABLED  # leader starts disabled
            fd = self._perf_event_open(attr, self._leader)
            if fd < 0:
                err = ctypes.get_errno()
                logger.debug(
                    "perf_event_open failed for %s: errno=%d (%s)",
                    _EVENTS[i][0], err, os.strerror(err),
                )
                return False
            self._fds.append(fd)
            if i == 0:
                self._leader = fd
        return True

    def close(self) -> None:
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds = []
        self._leader = -1

    # ── control (whole group via leader) ──────────────────────────────────────

    def reset(self) -> None:
        fcntl.ioctl(self._leader, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP)

    def enable(self) -> None:
        fcntl.ioctl(self._leader, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP)

    def disable(self) -> None:
        fcntl.ioctl(self._leader, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP)

    # ── read ──────────────────────────────────────────────────────────────────

    def read(self) -> CounterTotals:
        """Read the group leader. Layout (read_format = GROUP|ENABLED|RUNNING):

            u64 nr;
            u64 time_enabled;
            u64 time_running;
            u64 values[nr];   # in counter-creation order

        If the PMU multiplexed (time_running < time_enabled), scale up linearly.
        Any failure → all-None (degrade to ns-only).
        """
        n = len(_EVENTS)
        nwords = 3 + n
        try:
            raw = os.read(self._leader, 8 * nwords)
        except OSError:
            return CounterTotals(None, None, None)
        if len(raw) < 8 * nwords:
            return CounterTotals(None, None, None)

        words = [
            int.from_bytes(raw[i * 8:(i + 1) * 8], "little") for i in range(nwords)
        ]
        nr, time_enabled, time_running = words[0], words[1], words[2]
        values = words[3:3 + n]
        if nr != n:
            return CounterTotals(None, None, None)

        scale = 1.0
        if time_running > 0 and time_running < time_enabled:
            scale = time_enabled / time_running

        scaled = {
            name: int(round(values[i] * scale)) for i, (name, _) in enumerate(_EVENTS)
        }
        return CounterTotals(
            cycles=scaled["cycles"],
            instructions=scaled["instructions"],
            cache_misses=scaled["cache_misses"],
        )

    # ── context manager sugar ──────────────────────────────────────────────────

    def __enter__(self) -> "PerfCounters":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_counters() -> Optional[PerfCounters]:
    """Open a cycles/instructions/cache-misses group, or None if unavailable.

    None means: not Linux, no libc, unknown arch, or the kernel denied the
    syscall (paranoid setting / no PMU). The caller proceeds with ns-only timing.
    """
    if platform.system() != "Linux":
        return None
    nr = _SYSCALL_NR.get(platform.machine())
    if nr is None:
        return None
    libc = _libc()
    if libc is None:
        return None
    libc.syscall.restype = ctypes.c_int

    pc = PerfCounters(libc, nr)
    try:
        if not pc._open_group():
            pc.close()
            return None
    except Exception:  # noqa: BLE001 — never let counter setup break a run
        logger.debug("open_counters raised", exc_info=True)
        pc.close()
        return None
    return pc


__all__ = ["CounterTotals", "PerfCounters", "open_counters"]
