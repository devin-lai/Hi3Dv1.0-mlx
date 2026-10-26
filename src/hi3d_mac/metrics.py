"""Wall-clock and memory measurement on macOS.

Memory is the process *physical footprint* (``proc_pid_rusage``), the figure
Activity Monitor reports. On Apple silicon it includes Metal buffers in unified
memory, which ``ru_maxrss`` does not, so it covers both PyTorch-MPS and MLX.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import psutil

_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_RUSAGE_INFO_V4 = 4


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups "
            "ri_pageins ri_wired_size ri_resident_size ri_phys_footprint "
            "ri_proc_start_abstime ri_proc_exit_abstime ri_child_user_time "
            "ri_child_system_time ri_child_pkg_idle_wkups ri_child_interrupt_wkups "
            "ri_child_pageins ri_child_elapsed_abstime ri_diskio_bytesread "
            "ri_diskio_byteswritten ri_cpu_time_qos_default "
            "ri_cpu_time_qos_maintenance ri_cpu_time_qos_background "
            "ri_cpu_time_qos_utility ri_cpu_time_qos_legacy "
            "ri_cpu_time_qos_user_initiated ri_cpu_time_qos_user_interactive "
            "ri_billed_system_time ri_serviced_system_time ri_logical_writes "
            "ri_lifetime_max_phys_footprint ri_instructions ri_cycles "
            "ri_billed_energy ri_serviced_energy ri_interval_max_phys_footprint "
            "ri_runnable_time"
        ).split()
    ]


def _rusage(pid: int | None = None) -> _RusageInfoV4:
    info = _RusageInfoV4()
    if _libc.proc_pid_rusage(os.getpid() if pid is None else pid, _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    return info


def footprint_bytes() -> int:
    """Current physical footprint of this process."""
    return int(_rusage().ri_phys_footprint)


def lifetime_peak_bytes() -> int:
    """Largest physical footprint this process has had."""
    return int(_rusage().ri_lifetime_max_phys_footprint)


GB = 1024**3


@dataclass
class PhaseRecord:
    name: str
    seconds: float
    peak_gb: float
    end_gb: float
    tree_peak_gb: float


@dataclass
class Profiler:
    """Times named phases and samples the footprint peak inside each."""

    interval: float = 0.02
    phases: list[PhaseRecord] = field(default_factory=list)

    def __post_init__(self):
        self._peak = 0
        self._tree_peak = 0
        self._tree_lifetime_peak = 0
        self._process = psutil.Process()
        self._children: list[int] = []
        self._next_children_refresh = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while not self._stop.wait(self.interval):
            value = footprint_bytes()
            # Discover subprocesses at 5 Hz; sample their physical footprints at
            # the same 50 Hz as the parent. This includes the MPS conditioner.
            if time.monotonic() >= self._next_children_refresh:
                self._children = [p.pid for p in self._process.children(recursive=True)]
                self._next_children_refresh = time.monotonic() + 0.2
            tree = value
            for pid in self._children:
                try:
                    tree += int(_rusage(pid).ri_phys_footprint)
                except OSError:  # child exited between discovery and sampling
                    pass
            with self._lock:
                self._peak = max(self._peak, value)
                self._tree_peak = max(self._tree_peak, tree)
                self._tree_lifetime_peak = max(self._tree_lifetime_peak, tree)

    @contextmanager
    def phase(self, name: str, sync=None):
        with self._lock:
            self._peak = footprint_bytes()
            self._tree_peak = self._peak
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync is not None:
                sync()
            seconds = time.perf_counter() - start
            end = footprint_bytes()
            with self._lock:
                peak = max(self._peak, end)
                tree_peak = max(self._tree_peak, end)
            self.phases.append(PhaseRecord(name, seconds, peak / GB, end / GB, tree_peak / GB))

    def close(self):
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        totals: dict[str, float] = {}
        for record in self.phases:
            totals[record.name] = totals.get(record.name, 0.0) + record.seconds
        return {
            "phases": [record.__dict__ for record in self.phases],
            "seconds_by_phase": totals,
            "peak_footprint_gb": max((r.peak_gb for r in self.phases), default=0.0),
            "lifetime_peak_footprint_gb": lifetime_peak_bytes() / GB,
            "peak_process_tree_footprint_gb": self._tree_lifetime_peak / GB,
            "memory_note": "GiB; process tree is a sampled sum of parent and descendants' physical "
                           "footprints, conservatively counting any shared pages in each process. "
                           "Legacy peak_footprint_gb is parent-only.",
        }
