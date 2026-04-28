"""
Streaming activity estimator trained on your labeled CSVs (few-shot).

Supports:

- **JSON centroid model** (default from ``train_fewshot_activity.py``): L2 distance to
  per-class mean feature vectors; **numpy only**, no scikit-learn required.
- **Joblib sklearn** bundle ``{"model": clf, "meta": ...}`` if you train with
  ``--sklearn-out`` (optional RF when sklearn works in your environment).
"""

from __future__ import print_function

import json
import os
from collections import deque
from threading import Lock
from time import monotonic

import numpy as np

from har_imu.feature_window import (
    build_mag_series_from_samples,
    featurize_mag_series,
)
from har_imu.realtime_estimator import RealtimeActivityEstimator


def _softmax_neg_dist(dists_sq, temperature=1.0):
    """Higher score = closer (smaller distance)."""
    z = -np.asarray(dists_sq, dtype=np.float64) / max(float(temperature), 1e-9)
    z = z - np.max(z)
    e = np.exp(z)
    s = e / (np.sum(e) + 1e-12)
    return s


class FewShotStreamEstimator(object):
    """Thread-safe push + periodic recompute; ``snapshot()`` matches heuristic JSON shape."""

    def __init__(self, bundle, *, min_recompute_s=1.0, vote_len=5):
        meta = bundle.get("meta") or {}
        self.window_s = float(meta.get("window_s", 14.0))
        self.target_fs = float(meta.get("target_fs", 40.0))
        self.min_recompute_s = float(min_recompute_s)
        self.vote_len = int(vote_len)

        if "model" in bundle:
            self._kind = "sklearn"
            self._clf = bundle["model"]
        elif bundle.get("kind") == "centroid_l2":
            self._kind = "centroid"
            self._classes = [str(c) for c in bundle["classes"]]
            self._centroids = np.asarray(bundle["centroids"], dtype=np.float64)
            if self._centroids.ndim != 2:
                raise ValueError("centroids must be 2-D (n_classes x n_features)")
        else:
            raise ValueError("Unknown bundle: need sklearn 'model' or kind 'centroid_l2'")

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

            X = feat.reshape(1, -1)
            proba = None
            classes = None
            if self._kind == "sklearn":
                try:
                    proba = self._clf.predict_proba(X)[0]
                    classes = [str(c) for c in self._clf.classes_]
                    k = int(np.argmax(proba))
                    raw = classes[k]
                    conf = float(proba[k])
                except Exception:
                    raw = str(self._clf.predict(X)[0])
                    conf = 0.55
                    proba = None
                    classes = None
                detail0 = "fewshot_sklearn raw={} p={:.2f}".format(raw, conf)
            else:
                d = np.sum((self._centroids - feat) ** 2, axis=1)
                proba = _softmax_neg_dist(d, temperature=max(0.5 * float(np.median(d) + 1e-6), 1e-3))
                classes = self._classes
                k = int(np.argmax(proba))
                raw = self._classes[k]
                conf = float(proba[k])
                detail0 = "fewshot_centroid raw={} p={:.2f}".format(raw, conf)

            self._votes.append(raw)
            maj = RealtimeActivityEstimator._majority(list(self._votes))
            self.label = maj
            self.confidence = float(conf) if maj == raw else max(0.35, float(conf) * 0.85)
            detail = detail0
            if proba is not None and classes is not None:
                detail += " | " + " ".join(
                    "{}={:.2f}".format(str(c), float(p)) for c, p in zip(classes, proba)
                )
            self.detail = detail + " | vote={}".format(",".join(self._votes))
            self.window_samples_acc = len(mag_acc)
            self.updated_mono = now_mono

    def snapshot(self):
        with self._lock:
            backend = "fewshot_sklearn" if self._kind == "sklearn" else "fewshot_centroid"
            return {
                "label": self.label,
                "confidence": round(self.confidence, 3),
                "detail": self.detail,
                "window_samples_acc": self.window_samples_acc,
                "window_s": self.window_s,
                "backend": backend,
                "updated_age_s": round(monotonic() - self.updated_mono, 2)
                if self.updated_mono > 0
                else None,
            }


def load_activity_model(path):
    """Load JSON centroid model or joblib sklearn bundle."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    lower = path.lower()
    if lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        if bundle.get("kind") != "centroid_l2":
            raise RuntimeError("JSON model must have kind 'centroid_l2' (from train_fewshot_activity.py)")
        return bundle
    try:
        import joblib
    except ImportError as e:
        raise RuntimeError("joblib required to load .joblib activity model") from e
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise RuntimeError("Joblib file must contain dict with key 'model'")
    return bundle


# Backwards compatibility
FewShotRFStreamEstimator = FewShotStreamEstimator
load_fewshot_bundle = load_activity_model
