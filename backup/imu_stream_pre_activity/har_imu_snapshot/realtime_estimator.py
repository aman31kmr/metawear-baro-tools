"""
Near-realtime standing / walking / running from CORRECTED_ACC (linear g) + gyro magnitude.

Heuristic thresholds — tune for your mounting after field trials. Uses a sliding
board-time window and recomputes at most every ``min_recompute_s`` seconds.
"""

from __future__ import print_function

from collections import Counter, deque
from threading import Lock
from time import monotonic

import numpy as np


class RealtimeActivityEstimator(object):
    """Thread-safe push + periodic recompute; snapshot for JSON/UI."""

    def __init__(
        self,
        *,
        window_s=14.0,
        target_fs=40.0,
        min_recompute_s=1.0,
        vote_len=5,
    ):
        self.window_s = float(window_s)
        self.target_fs = float(target_fs)
        self.min_recompute_s = float(min_recompute_s)
        self.vote_len = int(vote_len)

        self._lock = Lock()
        self._acc = deque(maxlen=25000)  # (epoch_ms, x, y, z)
        self._gyro = deque(maxlen=25000)
        self._last_compute = 0.0
        self._votes = deque(maxlen=self.vote_len)

        self.label = "unknown"
        self.confidence = 0.0
        self.detail = ""
        self.window_samples_acc = 0
        self.updated_mono = 0.0

    def push(self, sensor, epoch_ms, x, y, z):
        with self._lock:
            if sensor == "acc":
                self._acc.append((int(epoch_ms), float(x), float(y), float(z)))
            elif sensor == "gyro":
                self._gyro.append((int(epoch_ms), float(x), float(y), float(z)))

    def maybe_recompute(self, now_mono=None):
        if now_mono is None:
            now_mono = monotonic()
        with self._lock:
            if now_mono - self._last_compute < self.min_recompute_s:
                return
            self._last_compute = now_mono

            acc = list(self._acc)
            gyro = list(self._gyro)
            if len(acc) < 32:
                self.label = "unknown"
                self.confidence = 0.0
                self.detail = "buffering acc…"
                self.window_samples_acc = len(acc)
                self.updated_mono = now_mono
                return

            epoch0 = acc[0][0]
            t_acc = np.array([(e - epoch0) * 1e-3 for e, _, _, _ in acc], dtype=np.float64)
            A = np.array([[r[1], r[2], r[3]] for r in acc], dtype=np.float64)

            t_max = t_acc[-1]
            t_min = max(0.0, t_max - self.window_s)
            m_acc = t_acc >= t_min
            t_acc = t_acc[m_acc]
            A = A[m_acc]
            if len(t_acc) < 24:
                self.label = "unknown"
                self.confidence = 0.0
                self.detail = "window too short"
                self.window_samples_acc = len(t_acc)
                self.updated_mono = now_mono
                return

            grid, mag_acc = self._resample_mag(t_acc, A, self.target_fs)
            if grid is None or len(mag_acc) < 24:
                self.label = "unknown"
                self.confidence = 0.0
                self.detail = "resample failed"
                self.window_samples_acc = len(t_acc)
                self.updated_mono = now_mono
                return

            mag_gyro = None
            if len(gyro) >= 8:
                tg = np.array([(e - epoch0) * 1e-3 for e, _, _, _ in gyro], dtype=np.float64)
                G = np.array([[r[1], r[2], r[3]] for r in gyro], dtype=np.float64)
                mg = np.linalg.norm(G, axis=1)
                mag_gyro = np.interp(grid, tg, mg)

            raw_label, conf, detail = self._classify(mag_acc, mag_gyro)
            self._votes.append(raw_label)
            maj = self._majority(list(self._votes))
            self.label = maj
            self.confidence = float(conf) if maj == raw_label else max(0.35, float(conf) * 0.85)
            self.detail = detail + " | vote={}".format(",".join(self._votes))
            self.window_samples_acc = len(mag_acc)
            self.updated_mono = now_mono

    def snapshot(self):
        with self._lock:
            return {
                "label": self.label,
                "confidence": round(self.confidence, 3),
                "detail": self.detail,
                "window_samples_acc": self.window_samples_acc,
                "window_s": self.window_s,
                "updated_age_s": round(monotonic() - self.updated_mono, 2)
                if self.updated_mono > 0
                else None,
            }

    @staticmethod
    def _resample_mag(t_acc, A, target_fs):
        """Uniform grid linear interp of ||linear_acc|| (g)."""
        t0, t1 = float(t_acc[0]), float(t_acc[-1])
        dur = t1 - t0
        if dur < 0.05:
            return None, None
        n = max(32, int(dur * target_fs) + 1)
        grid = np.linspace(t0, t1, n, dtype=np.float64)
        ax = np.interp(grid, t_acc, A[:, 0])
        ay = np.interp(grid, t_acc, A[:, 1])
        az = np.interp(grid, t_acc, A[:, 2])
        mag = np.sqrt(ax * ax + ay * ay + az * az)
        return grid, mag

    @staticmethod
    def _majority(labels):
        if not labels:
            return "unknown"
        c = Counter(labels)
        return c.most_common(1)[0][0]

    def _classify(self, mag_acc, mag_gyro):
        """
        mag_acc: linear acceleration magnitude (g), detrended for FFT.
        Heuristic tuned for MetaWear fusion linear acc scale.
        """
        m = mag_acc - np.mean(mag_acc)
        std_m = float(np.std(m))
        fs = self.target_fs

        spec = np.abs(np.fft.rfft(m)) ** 2
        freqs = np.fft.rfftfreq(len(m), d=1.0 / fs)
        mask = (freqs >= 0.65) & (freqs <= 3.8)
        if not np.any(mask):
            return "standing", 0.4, "no spectrum"

        band = spec[mask]
        freqs_b = freqs[mask]
        step_power = float(np.sum(band))
        total_power = float(np.sum(spec)) + 1e-9
        ratio = step_power / total_power
        pk = int(np.argmax(band))
        peak_f = float(freqs_b[pk])

        gmean = float(np.mean(mag_gyro)) if mag_gyro is not None and len(mag_gyro) == len(m) else 0.0

        # --- thresholds (iterate later with your own labeled clips) ---
        stand_std = 0.055
        walk_std_lo = 0.09
        run_std = 0.28
        walk_ratio = 0.14
        run_ratio = 0.22

        detail_bits = [
            "std={:.3f}".format(std_m),
            "step_ratio={:.2f}".format(ratio),
            "peak_f={:.2f}Hz".format(peak_f),
            "gyro_mean={:.1f}".format(gmean),
        ]
        detail = " ".join(detail_bits)

        # running: high dynamics or high cadence + energy
        if std_m >= run_std or (std_m >= 0.16 and peak_f >= 2.05 and ratio >= run_ratio):
            raw = "running"
            conf = min(1.0, 0.5 + 0.5 * min(1.0, (std_m - run_std + 0.05) / 0.35))
        # standing: very quiet spectrum
        elif std_m < stand_std and ratio < walk_ratio:
            raw = "standing"
            conf = min(1.0, 0.45 + 0.55 * (1.0 - std_m / stand_std))
        # walking: periodic band + moderate std
        elif ratio >= walk_ratio and walk_std_lo <= std_m < run_std and 0.75 <= peak_f <= 2.45:
            raw = "walking"
            conf = min(1.0, 0.4 + 0.6 * min(1.0, ratio / 0.45))
        elif std_m >= walk_std_lo and ratio >= (walk_ratio * 0.85) and peak_f < 2.35:
            raw = "walking"
            conf = 0.55
        elif gmean > 120.0 and std_m >= 0.12:
            raw = "running"
            conf = 0.5
        else:
            raw = "standing"
            conf = 0.4

        return raw, conf, detail
