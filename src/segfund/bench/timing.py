"""GPU timing with clock verification.

The rules are ported from the author's tritonkv benchmark (github.com/PS12007/tritonkv),
which found a 12x measurement artifact from timing a laptop GPU at idle clocks:

  * the RTX 5060 Laptop idles near 285 MHz and boosts to ~3090 MHz, so a kernel
    timed before the clock ramps is timed on a different GPU;
  * sample clocks in the background (one long-lived `nvidia-smi -lms` process);
  * ramp the GPU with the workload itself before timing;
  * a timing row is QUOTABLE only if every SM-clock sample in its window is
    >= 70% of max, the GPU is below 87 C, the window holds >= 4 clock samples,
    and the timing IQR is <= 5% of the median.
Non-quotable rows are still recorded, with the reason, never silently dropped.

Times come from CUDA events around each launch (device time, not Python time).
Reported: median, 95% bootstrap CI of the median, IQR as a fraction of the median.
"""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

SMI_QUERY = "clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu"
SMI_KEYS = ("sm_mhz", "mem_mhz", "temp_c", "power_w", "util_pct")

CLOCK_BOOST_FLOOR_FRAC = 0.70
CLOCK_TARGET_FRAC = 0.80
CLOCK_TEMP_LIMIT_C = 87.0
MAX_IQR_FRAC = 0.05
MIN_CLOCK_SAMPLES = 4
MIN_SAMPLING_SECONDS = 1.5


def _smi(fields: str) -> list[str] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    except Exception:
        return None


def max_sm_clock() -> float | None:
    v = _smi("clocks.max.sm")
    return float(v[0]) if v else None


def gpu_static() -> dict:
    keys = ("name", "driver_version", "clocks.max.sm", "power.limit")
    v = _smi(",".join(keys))
    return dict(zip(keys, v)) if v else {}


class ClockMonitor:
    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.max_sm = max_sm_clock()
        self.samples: list[tuple[float, dict]] = []
        self.proc = None
        self.available = False
        self._stop = threading.Event()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self) -> bool:
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={SMI_QUERY}", "--format=csv,noheader,nounits",
                 "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except Exception:
            return False
        threading.Thread(target=self._read, daemon=True).start()
        time.sleep(0.6)
        self.available = bool(self.samples)
        return self.available

    def _read(self):
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != len(SMI_KEYS):
                continue
            try:
                self.samples.append((time.time(), {k: float(v) for k, v in zip(SMI_KEYS, parts)}))
            except ValueError:
                continue

    def latest(self) -> dict | None:
        return self.samples[-1][1] if self.samples else None

    def window(self, t0: float, t1: float) -> dict:
        vals = [s for (t, s) in self.samples if t0 <= t <= t1]
        out = {"n_samples": len(vals)}
        if not vals:
            out.update(boosted=False, thermal_ok=False, enough_samples=False, stable=False)
            return out
        for k in SMI_KEYS:
            xs = [v[k] for v in vals]
            out[f"{k}_min"], out[f"{k}_max"], out[f"{k}_mean"] = min(xs), max(xs), statistics.fmean(xs)
        out["boosted"] = bool(self.max_sm and out["sm_mhz_min"] >= CLOCK_BOOST_FLOOR_FRAC * self.max_sm)
        out["thermal_ok"] = out["temp_c_max"] < CLOCK_TEMP_LIMIT_C
        out["enough_samples"] = out["n_samples"] >= MIN_CLOCK_SAMPLES
        out["stable"] = bool(out["boosted"] and out["thermal_ok"] and out["enough_samples"])
        return out

    def stop(self):
        self._stop.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass


@dataclass
class Timing:
    label: str
    n_reps: int
    median_ms: float
    ci95_ms: tuple[float, float]
    iqr_frac: float
    clocks: dict
    quotable: bool
    reasons: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def _bootstrap_median_ci(xs: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(xs, size=(n_boot, xs.size), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def ramp(fn, monitor: ClockMonitor | None, max_seconds: float = 5.0) -> dict:
    """Run the workload until the SM clock reaches CLOCK_TARGET_FRAC of max (or time out)."""
    import torch

    t0 = time.time()
    while time.time() - t0 < max_seconds:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s = monitor.latest() if monitor and monitor.available else None
        if s and monitor.max_sm and s["sm_mhz"] >= CLOCK_TARGET_FRAC * monitor.max_sm and time.time() - t0 > 0.5:
            break
    s = monitor.latest() if monitor and monitor.available else None
    return {"seconds": time.time() - t0, "sm_mhz_after": s["sm_mhz"] if s else None}


def time_gpu(label: str, fn, monitor: ClockMonitor | None, min_reps: int = 20, min_seconds: float = MIN_SAMPLING_SECONDS) -> Timing:
    """Time fn() with CUDA events until both min_reps and min_seconds are reached."""
    import torch

    ramp(fn, monitor)
    times = []
    t_start = time.time()
    while len(times) < min_reps or time.time() - t_start < min_seconds:
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
        if len(times) > 5000:
            break
    t_end = time.time()
    xs = np.asarray(times)
    med = float(np.median(xs))
    q1, q3 = np.percentile(xs, [25, 75])
    iqr_frac = float((q3 - q1) / med) if med > 0 else float("inf")
    clocks = monitor.window(t_start, t_end) if monitor and monitor.available else {"stable": False, "n_samples": 0}
    reasons = []
    if not (monitor and monitor.available):
        reasons.append("no clock monitor")
    else:
        if not clocks.get("boosted"):
            reasons.append("SM clock below 70% of max in window")
        if not clocks.get("thermal_ok"):
            reasons.append("temperature limit")
        if not clocks.get("enough_samples"):
            reasons.append("fewer than 4 clock samples")
    if iqr_frac > MAX_IQR_FRAC:
        reasons.append(f"IQR {iqr_frac:.1%} > 5% of median")
    return Timing(label, xs.size, med, _bootstrap_median_ci(xs), iqr_frac, clocks, not reasons, reasons)
