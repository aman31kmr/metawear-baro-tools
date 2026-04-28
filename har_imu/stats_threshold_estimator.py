"""
Streaming stats-threshold activity estimator (standing / walking / running).

Primary feature: SD of acceleration magnitude over a sliding time window.
Thresholds come from ``train_stats_activity.py``.
"""

from __future__ import print_function

import json
import os
from collections import deque
from threading import Lock
from time import monotonic

import numpy as np

from har_imu.realtime_estimator import RealtimeActivityEstimator


class StatsThresholdStreamEstimator(object):
    def __init__(self, model, *, min_recompute_s=1.0, vote_len=5):
        if model.get("kind") != "stats_threshold_v1":
            raise ValueError("Expected stats_threshold_v1 model")
        self.window_s = float(model.get("window_s", 2.0))
        self.min_recompute_s = float(min_recompute_s)
        self.vote_len = int(vote_len)
        th = model.get("thresholds") or {}
        self.th_stand_walk = float(th.get("stand_walk_acc_sd"))
        self.th_walk_run = float(th.get("walk_run_acc_sd"))

        self._lock = Lock()
        self._acc = deque(maxlen=25000)  # (epoch_ms, mag)
        self._gyro = deque(maxlen=25000)
        self._last_compute = 0.0
        self._votes = deque(maxlen=self.vote_len)

        self.label = "unknown"
        self.confidence = 0.0
        self.detail = ""
        self.window_samples_acc = 0
        self.updated_mono = 0.0

    def push(self, sensor, epoch_ms, x, y, z):
        mag = float((float(x) ** 2 + float(y) ** 2 + float(z) ** 2) ** 0.5)
        with self._lock:
            if sensor == "acc":
                self._acc.append((int(epoch_ms), mag))
            elif sensor == "gyro":
                self._gyro.append((int(epoch_ms), mag))

    def maybe_recompute(self, now_mono=None):
        if now_mono is None:
            now_mono = monotonic()
        with self._lock:
            if now_mono - self._last_compute < self.min_recompute_s:
                return
            self._last_compute = now_mono

            acc = list(self._acc)
            if len(acc) < 24:
                self.label = "unknown"
                self.confidence = 0.0
                self.detail = "buffering acc…"
                self.window_samples_acc = len(acc)
                self.updated_mono = now_mono
                return

            t_max = acc[-1][0]
            t_min = t_max - int(self.window_s * 1000.0)
            xs = [v for (t, v) in acc if t >= t_min]
            if len(xs) < 24:
                self.label = "unknown"
                self.confidence = 0.0
                self.detail = "window too short"
                self.window_samples_acc = len(xs)
                self.updated_mono = now_mono
                return

            x = np.asarray(xs, dtype=np.float64)
            sd = float(x.std(ddof=0))

            if sd < self.th_stand_walk:
                raw = "standing"
                # Map distance to threshold into a loose confidence
                conf = 0.55 + 0.45 * max(0.0, min(1.0, (self.th_stand_walk - sd) / max(self.th_stand_walk, 1e-6)))
            elif sd < self.th_walk_run:
                raw = "walking"
                mid = 0.5 * (self.th_stand_walk + self.th_walk_run)
                span = max(1e-6, 0.5 * (self.th_walk_run - self.th_stand_walk))
                conf = 0.55 + 0.45 * max(0.0, 1.0 - abs(sd - mid) / span)
            else:
                raw = "running"
                conf = 0.55 + 0.45 * max(0.0, min(1.0, (sd - self.th_walk_run) / max(self.th_walk_run, 1e-6)))

            self._votes.append(raw)
            maj = RealtimeActivityEstimator._majority(list(self._votes))
            self.label = maj
            self.confidence = float(conf) if maj == raw else max(0.35, float(conf) * 0.85)
            self.detail = (
                "stats acc_sd={:.4f} thr=({:.4f},{:.4f}) | vote={}".format(
                    sd, self.th_stand_walk, self.th_walk_run, ",".join(self._votes)
                )
            )
            self.window_samples_acc = len(xs)
            self.updated_mono = now_mono

    def snapshot(self):
        with self._lock:
            return {
                "label": self.label,
                "confidence": round(self.confidence, 3),
                "detail": self.detail,
                "window_samples_acc": self.window_samples_acc,
                "window_s": self.window_s,
                "backend": "stats_threshold",
                "updated_age_s": round(monotonic() - self.updated_mono, 2)
                if self.updated_mono > 0
                else None,
            }


def load_stats_model(path):
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    if not isinstance(model, dict) or model.get("kind") != "stats_threshold_v1":
        raise RuntimeError("Not a stats_threshold_v1 JSON model: {}".format(path))
    return model

