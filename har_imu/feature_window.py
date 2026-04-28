"""
Fixed-length feature vectors from CORRECTED_ACC + CORRECTED_GYRO windows.

Shared by ``train_fewshot_activity.py`` (offline) and streaming estimators (online).
Uses the same resampling grid as ``RealtimeActivityEstimator``.
"""

from __future__ import print_function

import numpy as np

from har_imu.realtime_estimator import RealtimeActivityEstimator


def featurize_mag_series(mag_acc, mag_gyro, target_fs):
    """
    ``mag_acc`` and ``mag_gyro`` are same-length arrays on a uniform time grid (see
    ``RealtimeActivityEstimator._resample_mag`` + gyro interp). Returns a 1-D float
    vector or None if insufficient data.
    """
    if mag_acc is None or len(mag_acc) < 24:
        return None
    m = np.asarray(mag_acc, dtype=np.float64) - np.mean(mag_acc)
    std_m = float(np.std(m))
    fs = float(target_fs)

    spec = np.abs(np.fft.rfft(m)) ** 2
    freqs = np.fft.rfftfreq(len(m), d=1.0 / fs)
    mask = (freqs >= 0.65) & (freqs <= 3.8)
    if not np.any(mask):
        band = np.array([0.0])
        freqs_b = np.array([1.0])
    else:
        band = spec[mask]
        freqs_b = freqs[mask]

    step_power = float(np.sum(band))
    total_power = float(np.sum(spec)) + 1e-9
    ratio = step_power / total_power
    pk = int(np.argmax(band)) if len(band) else 0
    peak_f = float(freqs_b[pk]) if len(freqs_b) else 0.0

    mag_mean = float(np.mean(mag_acc))
    mag_max = float(np.max(mag_acc))
    mag_min = float(np.min(mag_acc))

    if mag_gyro is not None and len(mag_gyro) == len(mag_acc):
        gmean = float(np.mean(mag_gyro))
        gstd = float(np.std(mag_gyro))
        gmax = float(np.max(mag_gyro))
    else:
        gmean = gstd = gmax = 0.0

    # Compact spectrum shape: log-energy in 3 sub-bands (Hz)
    def band_energy(lo, hi):
        m2 = (freqs >= lo) & (freqs <= hi)
        if not np.any(m2):
            return 0.0
        return float(np.log(np.sum(spec[m2]) + 1e-12))

    e_low = band_energy(0.5, 1.1)
    e_mid = band_energy(1.1, 2.2)
    e_hi = band_energy(2.2, 4.0)

    vec = np.array(
        [
            std_m,
            ratio,
            peak_f,
            gmean,
            gstd,
            gmax,
            mag_mean,
            mag_max,
            mag_min,
            e_low,
            e_mid,
            e_hi,
        ],
        dtype=np.float64,
    )
    return vec


def build_mag_series_from_samples(acc_samples, gyro_samples, target_fs):
    """
    acc_samples / gyro_samples: list of (epoch_ms, x, y, z), sorted by epoch.
    Returns (grid, mag_acc, mag_gyro) or (None, None, None).
    """
    if len(acc_samples) < 8:
        return None, None, None
    epoch0 = int(acc_samples[0][0])
    t_acc = np.array([(e - epoch0) * 1e-3 for e, _, _, _ in acc_samples], dtype=np.float64)
    A = np.array([[r[1], r[2], r[3]] for r in acc_samples], dtype=np.float64)
    grid, mag_acc = RealtimeActivityEstimator._resample_mag(t_acc, A, target_fs)
    if grid is None or mag_acc is None or len(mag_acc) < 24:
        return None, None, None

    mag_gyro = None
    if len(gyro_samples) >= 8:
        tg = np.array([(e - epoch0) * 1e-3 for e, _, _, _ in gyro_samples], dtype=np.float64)
        G = np.array([[r[1], r[2], r[3]] for r in gyro_samples], dtype=np.float64)
        mg = np.linalg.norm(G, axis=1)
        mag_gyro = np.interp(grid, tg, mg)
    return grid, mag_acc, mag_gyro


def window_slice_by_time(acc_samples, gyro_samples, t0_s, t1_s, epoch0_ms):
    """Keep samples with epoch in [t0_s, t1_s] relative to epoch0_ms (seconds)."""
    lo = epoch0_ms + int(t0_s * 1000)
    hi = epoch0_ms + int(t1_s * 1000)
    acc_w = [r for r in acc_samples if lo <= int(r[0]) <= hi]
    gyro_w = [r for r in gyro_samples if lo <= int(r[0]) <= hi]
    return acc_w, gyro_w


def feature_names():
    return [
        "acc_mag_std",
        "step_band_power_ratio",
        "step_band_peak_hz",
        "gyro_mag_mean",
        "gyro_mag_std",
        "gyro_mag_max",
        "acc_mag_mean",
        "acc_mag_max",
        "acc_mag_min",
        "log_spec_0p5_1p1",
        "log_spec_1p1_2p2",
        "log_spec_2p2_4",
    ]
