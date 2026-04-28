"""
Optional NLI zero-shot activity labels: turns the last IMU window into a short English
summary, then runs HuggingFace ``zero-shot-classification`` (no training on your CSVs).

Install (large download on first run):
  pip install transformers torch sentencepiece

Override model:
  export HAR_NLI_MODEL=typeform/distilbert-base-uncased-mnli
"""

from __future__ import print_function

import os
from collections import deque
from threading import Lock
from time import monotonic

import numpy as np

from har_imu.feature_window import (
    build_mag_series_from_samples,
    featurize_mag_series,
    feature_names,
)
from har_imu.realtime_estimator import RealtimeActivityEstimator


def _feature_dict(vec):
    names = feature_names()
    out = {}
    for i, n in enumerate(names):
        out[n] = float(vec[i]) if i < len(vec) else 0.0
    return out


def _summary_text(vec):
    d = _feature_dict(vec)
    return (
        "Wrist sensor summary for the last seconds: "
        "acceleration magnitude varies with standard deviation {acc_mag_std:.3f} g; "
        "cadence-related band holds about {step_band_power_ratio:.0%} of spectral energy "
        "with strongest rhythm near {step_band_peak_hz:.2f} Hz; "
        "gyroscope magnitude averages {gyro_mag_mean:.0f} with std {gyro_mag_std:.0f}."
    ).format(**d)


_CANDIDATE_LABELS = [
    "standing still or only shifting weight slightly",
    "walking with a regular step cadence",
    "running or jogging with vigorous arm and body motion",
]

_LABEL_TO_ACTIVITY = {
    "standing still or only shifting weight slightly": "standing",
    "walking with a regular step cadence": "walking",
    "running or jogging with vigorous arm and body motion": "running",
}


class NLIZeroShotStreamEstimator(object):
    def __init__(
        self,
        *,
        window_s=14.0,
        target_fs=40.0,
        min_recompute_s=1.5,
        vote_len=5,
        model_name=None,
    ):
        self.window_s = float(window_s)
        self.target_fs = float(target_fs)
        self.min_recompute_s = float(min_recompute_s)
        self.vote_len = int(vote_len)
        self._model_name = model_name or os.environ.get(
            "HAR_NLI_MODEL", "typeform/distilbert-base-uncased-mnli"
        )
        self._pipe = None
        self._pipe_error = None

        self._lock = Lock()
        self._acc = deque(maxlen=25000)
        self._gyro = deque(maxlen=25000)
        self._last_compute = 0.0
        self._votes = deque(maxlen=self.vote_len)

        self.label = "unknown"
        self.confidence = 0.0
        self.detail = ""
        self.window_samples_acc = 0
        self.updated_mono = 0.0

    def _get_pipeline(self):
        if self._pipe is not None or self._pipe_error is not None:
            return self._pipe, self._pipe_error
        try:
            from transformers import pipeline
        except ImportError as e:
            self._pipe_error = (
                "transformers not installed. pip install transformers torch sentencepiece"
            )
            return None, self._pipe_error
        try:
            self._pipe = pipeline(
                "zero-shot-classification",
                model=self._model_name,
            )
        except Exception as e:
            self._pipe_error = str(e)
            return None, self._pipe_error
        return self._pipe, None

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
            t_max = float(t_acc[-1])
            t0 = max(0.0, t_max - self.window_s)
            acc_list = [r for r in acc if t0 <= (r[0] - epoch0) * 1e-3 <= t_max]
            gyro_list = [g for g in gyro if t0 <= (g[0] - epoch0) * 1e-3 <= t_max]

            if len(acc_list) < 24:
                self.label = "unknown"
                self.confidence = 0.0
                self.detail = "window too short"
                self.window_samples_acc = len(acc_list)
                self.updated_mono = now_mono
                return

            _, mag_acc, mag_gyro = build_mag_series_from_samples(acc_list, gyro_list, self.target_fs)
            feat = featurize_mag_series(mag_acc, mag_gyro, self.target_fs)
            if feat is None:
                self.label = "unknown"
                self.confidence = 0.0
                self.detail = "feature extract failed"
                self.window_samples_acc = len(acc_list)
                self.updated_mono = now_mono
                return

            pipe, err = self._get_pipeline()
            if pipe is None:
                self.label = "unknown"
                self.confidence = 0.0
                self.detail = "nli: {}".format(err or "init failed")
                self.window_samples_acc = len(mag_acc)
                self.updated_mono = now_mono
                return

            text = _summary_text(feat)
            try:
                zr = pipe(
                    text,
                    candidate_labels=_CANDIDATE_LABELS,
                    hypothesis_template="The person is {}.",
                    multi_label=False,
                )
                top_l = zr["labels"][0]
                top_s = float(zr["scores"][0])
                raw = _LABEL_TO_ACTIVITY.get(top_l, "unknown")
            except Exception as ex:
                self.label = "unknown"
                self.confidence = 0.0
                self.detail = "nli inference error: {}".format(ex)
                self.window_samples_acc = len(mag_acc)
                self.updated_mono = now_mono
                return

            self._votes.append(raw)
            maj = RealtimeActivityEstimator._majority(list(self._votes))
            self.label = maj
            self.confidence = float(top_s) if maj == raw else max(0.35, float(top_s) * 0.85)
            self.detail = (
                "nli_zero model={} raw={} | {}".format(self._model_name, raw, text[:120])
                + " | vote={}".format(",".join(self._votes))
            )
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
                "backend": "nli_zero_shot",
                "updated_age_s": round(monotonic() - self.updated_mono, 2)
                if self.updated_mono > 0
                else None,
            }
