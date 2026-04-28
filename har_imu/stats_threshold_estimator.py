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
    def __init__(
        self,
        model,
        *,
        min_recompute_s=1.0,
        vote_len=5,
        # Spike immunity + persistence controls:
        # - clip_quantiles: winsorize within-window magnitudes before SD
        # - min_state_s: minimum dwell time before switching label
        # - require_streak: require raw label to persist across N recomputes to switch
        clip_quantiles=(5.0, 95.0),
        min_state_s=2.5,
        require_streak=2,
    ):
        if model.get("kind") != "stats_threshold_v1":
            raise ValueError("Expected stats_threshold_v1 model")
        self.window_s = float(model.get("window_s", 2.0))
        self.min_recompute_s = float(min_recompute_s)
        self.vote_len = int(vote_len)
        self.clip_q_lo = float(clip_quantiles[0])
        self.clip_q_hi = float(clip_quantiles[1])
        self.min_state_s = float(min_state_s)
        self.require_streak = int(require_streak)
        th = model.get("thresholds") or {}
        self.th_stand_walk = float(th.get("stand_walk_acc_sd"))
        self.th_walk_run = float(th.get("walk_run_acc_sd"))

        self._lock = Lock()
        self._acc = deque(maxlen=25000)  # (epoch_ms, mag)
        self._gyro = deque(maxlen=25000)
        self._last_compute = 0.0
        self._votes = deque(maxlen=self.vote_len)
        self._raw_streak_label = None
        self._raw_streak_n = 0
        self._state_since_mono = 0.0

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
            # Winsorize (clip) to make SD robust to short spikes/outliers.
            # This is key for \"spike immunity\" without losing responsiveness.
            if x.size >= 16 and self.clip_q_hi > self.clip_q_lo:
                lo = float(np.percentile(x, self.clip_q_lo))
                hi = float(np.percentile(x, self.clip_q_hi))
                if hi > lo:
                    x = np.clip(x, lo, hi)
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

            # Persistence: only allow label switch if raw persists and current label has aged enough.
            # This prevents brief spikes from flipping the state.
            if raw == self._raw_streak_label:
                self._raw_streak_n += 1
            else:
                self._raw_streak_label = raw
                self._raw_streak_n = 1

            self._votes.append(raw)
            maj = RealtimeActivityEstimator._majority(list(self._votes))
            # Candidate next label prefers majority vote, but gated by streak + dwell time.
            next_label = maj
            cur = self.label
            can_switch = (
                (cur in ("unknown", "off") or (now_mono - self._state_since_mono) >= self.min_state_s)
                and self._raw_streak_n >= self.require_streak
            )
            if cur != next_label and not can_switch:
                next_label = cur
            if next_label != self.label:
                self._state_since_mono = now_mono
            self.label = next_label
            self.confidence = float(conf) if maj == raw else max(0.35, float(conf) * 0.85)
            self.detail = (
                "stats acc_sd={:.4f} thr=({:.4f},{:.4f}) clip=p{:.0f}-p{:.0f} streak={} dwell={:.1f}s | vote={}".format(
                    sd,
                    self.th_stand_walk,
                    self.th_walk_run,
                    self.clip_q_lo,
                    self.clip_q_hi,
                    self._raw_streak_n,
                    now_mono - (self._state_since_mono or now_mono),
                    ",".join(self._votes),
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

