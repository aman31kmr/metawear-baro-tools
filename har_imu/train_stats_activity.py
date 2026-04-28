#!/usr/bin/env python3
"""
Train a simple stats-threshold activity model from labeled session CSVs.

This model is intentionally lightweight: it uses windowed **acc magnitude SD**
as the primary discriminator (standing < walking < running), and can optionally
store a secondary gyro threshold for edge cases.

Output is a JSON file consumed by ``--activity-backend stats``.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import sys

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from imu_csv_format import parse_imu_csv_row  # noqa: E402


def _iter_manifest_files(labeled_root):
    for name in sorted(os.listdir(labeled_root)):
        if not name.startswith("session_"):
            continue
        sess = os.path.join(labeled_root, name)
        man_path = os.path.join(sess, "manifest.jsonl")
        if not os.path.isfile(man_path):
            continue
        with open(man_path, "r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                csv_name = rec.get("file")
                label = rec.get("activity")
                if not csv_name or not label:
                    continue
                csv_path = os.path.join(sess, csv_name)
                if os.path.isfile(csv_path):
                    yield csv_path, str(label).strip().lower()


def _load_mag_series(csv_path):
    """Return (activity, acc_mag_samples, gyro_mag_samples) with epoch_ms timestamps."""
    acc, gyro = [], []  # list of (epoch_ms, mag)
    activity = None
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            return None, acc, gyro
        for parts in r:
            row = parse_imu_csv_row(parts)
            if not row or row.get("epoch_ms") is None:
                continue
            e = int(row["epoch_ms"])
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            mag = float((x * x + y * y + z * z) ** 0.5)
            if row["sensor"] == "acc":
                acc.append((e, mag))
            elif row["sensor"] == "gyro":
                gyro.append((e, mag))
            if activity is None and row.get("activity"):
                activity = str(row["activity"]).strip().lower()
    return activity, acc, gyro


def _window_stats(samples, start_ms, end_ms):
    xs = [v for (t, v) in samples if start_ms <= t <= end_ms]
    if len(xs) < 12:
        return None
    x = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=0)),
        "p95": float(np.percentile(x, 95)),
    }


def collect_window_sds(labeled_root, *, window_s=2.0, stride_s=1.0):
    """
    Return dict: activity -> {'acc_sd': [...], 'gyro_mean': [...], 'gyro_sd': [...]}
    from sliding windows.
    """
    out = {}
    for act in ("standing", "walking", "running"):
        out[act] = {"acc_sd": [], "gyro_mean": [], "gyro_sd": []}

    items = list(_iter_manifest_files(labeled_root))
    it = tqdm(items, desc="CSV segments", unit="file") if tqdm is not None else items

    win_ms = int(float(window_s) * 1000)
    stride_ms = int(float(stride_s) * 1000)

    for csv_path, label in it:
        activity, acc, gyro = _load_mag_series(csv_path)
        act = (activity or label or "").strip().lower()
        if act not in out:
            continue
        if len(acc) < 24:
            continue
        acc.sort(key=lambda x: x[0])
        gyro.sort(key=lambda x: x[0])
        t0 = acc[0][0]
        t1 = acc[-1][0]
        cur = t0
        while cur + win_ms <= t1:
            a = _window_stats(acc, cur, cur + win_ms)
            g = _window_stats(gyro, cur, cur + win_ms) if gyro else None
            if a is not None:
                out[act]["acc_sd"].append(float(a["sd"]))
            if g is not None:
                out[act]["gyro_mean"].append(float(g["mean"]))
                out[act]["gyro_sd"].append(float(g["sd"]))
            cur += stride_ms
    return out


def _pick_threshold(lo_vals, hi_vals):
    """Pick a robust midpoint threshold separating lo vs hi distributions."""
    if not lo_vals or not hi_vals:
        return None
    lo = np.asarray(lo_vals, dtype=np.float64)
    hi = np.asarray(hi_vals, dtype=np.float64)
    # Robust centers
    c_lo = float(np.median(lo))
    c_hi = float(np.median(hi))
    if c_hi <= c_lo:
        return float(np.mean([c_lo, c_hi]))
    return float((c_lo + c_hi) * 0.5)


def _pick_threshold_quantile(lo_vals, hi_vals, *, lo_q=90.0, hi_q=10.0):
    """
    Quantile-based boundary, biased toward earlier detection of the high-activity class.

    Example: for walk->run, use midpoint( p90(walk), p10(run) ) which tends to be
    lower than midpoint(medians), so it flips to running sooner.
    """
    if not lo_vals or not hi_vals:
        return None
    lo = np.asarray(lo_vals, dtype=np.float64)
    hi = np.asarray(hi_vals, dtype=np.float64)
    a = float(np.percentile(lo, float(lo_q)))
    b = float(np.percentile(hi, float(hi_q)))
    return float(0.5 * (a + b))


def main():
    ap = argparse.ArgumentParser(description="Train stats-threshold activity model from labeled_data/")
    ap.add_argument(
        "--labeled-root",
        default="./labeled_data",
        help="Directory containing session_* folders with manifest.jsonl",
    )
    ap.add_argument("--window-s", type=float, default=2.0, help="Window length in seconds")
    ap.add_argument("--stride-s", type=float, default=1.0, help="Stride between windows")
    ap.add_argument(
        "--walk-run-lo-q",
        type=float,
        default=90.0,
        help="Quantile of walking acc_sd to use for walk->run threshold (default: 90)",
    )
    ap.add_argument(
        "--walk-run-hi-q",
        type=float,
        default=10.0,
        help="Quantile of running acc_sd to use for walk->run threshold (default: 10)",
    )
    ap.add_argument(
        "--out",
        default="./models/activity_stats.json",
        help="Output JSON path",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.labeled_root)
    d = collect_window_sds(root, window_s=args.window_s, stride_s=args.stride_s)

    thr_stand_walk = _pick_threshold(d["standing"]["acc_sd"], d["walking"]["acc_sd"])
    # Bias earlier running detection using quantiles by default (p90 walk vs p10 run).
    thr_walk_run = _pick_threshold_quantile(
        d["walking"]["acc_sd"],
        d["running"]["acc_sd"],
        lo_q=args.walk_run_lo_q,
        hi_q=args.walk_run_hi_q,
    )

    if thr_stand_walk is None or thr_walk_run is None:
        raise SystemExit("Not enough windows to compute thresholds; record more data.")

    model = {
        "version": 1,
        "kind": "stats_threshold_v1",
        "primary_feature": "acc_mag_sd",
        "window_s": float(args.window_s),
        "stride_s": float(args.stride_s),
        "thresholds": {
            "stand_walk_acc_sd": float(thr_stand_walk),
            "walk_run_acc_sd": float(thr_walk_run),
        },
        "calibration": {
            "standing_acc_sd_median": float(np.median(d["standing"]["acc_sd"])) if d["standing"]["acc_sd"] else None,
            "walking_acc_sd_median": float(np.median(d["walking"]["acc_sd"])) if d["walking"]["acc_sd"] else None,
            "running_acc_sd_median": float(np.median(d["running"]["acc_sd"])) if d["running"]["acc_sd"] else None,
            "n_windows": {
                "standing": len(d["standing"]["acc_sd"]),
                "walking": len(d["walking"]["acc_sd"]),
                "running": len(d["running"]["acc_sd"]),
            },
            "walk_run_quantiles": {
                "walking_lo_q": float(args.walk_run_lo_q),
                "running_hi_q": float(args.walk_run_hi_q),
            },
        },
    }

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    print("Wrote stats model:", out_path, flush=True)
    print("Threshold stand->walk (acc_sd):", model["thresholds"]["stand_walk_acc_sd"], flush=True)
    print("Threshold walk->run (acc_sd):", model["thresholds"]["walk_run_acc_sd"], flush=True)


if __name__ == "__main__":
    main()

