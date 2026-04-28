#!/usr/bin/env python3
"""
Train a few-shot activity model on labeled session CSVs (record lab layout).

Default output is a **JSON centroid** classifier (numpy only, no scikit-learn).
Optionally also writes a RandomForest **joblib** if sklearn imports cleanly.

Reads each ``session_*/manifest.jsonl`` under ``--labeled-root``, loads referenced
CSVs, slices sliding windows, builds features (``har_imu.feature_window``).

Stream with::

  python3 metawear_imu_stream.py <MAC> --webui --activity --activity-backend fewshot \\
      --activity-model ./models/activity_fewshot.json --csv ./run.csv
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

from har_imu.feature_window import (  # noqa: E402
    build_mag_series_from_samples,
    feature_names,
    featurize_mag_series,
    window_slice_by_time,
)


def _load_csv_streams(path):
    acc, gyro = [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        try:
            next(r)
        except StopIteration:
            return acc, gyro
        for parts in r:
            row = parse_imu_csv_row(parts)
            if not row or row.get("epoch_ms") is None:
                continue
            tup = (int(row["epoch_ms"]), float(row["x"]), float(row["y"]), float(row["z"]))
            if row["sensor"] == "acc":
                acc.append(tup)
            elif row["sensor"] == "gyro":
                gyro.append(tup)
    acc.sort(key=lambda x: x[0])
    gyro.sort(key=lambda x: x[0])
    return acc, gyro


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


def collect_windows(
    labeled_root,
    *,
    window_s=2.0,
    stride_s=1.0,
    target_fs=40.0,
):
    X_list, y_list = [], []
    files = list(_iter_manifest_files(labeled_root))
    if not files:
        raise SystemExit("No manifest.jsonl + CSV pairs under {!r}".format(labeled_root))

    iterator = files
    if tqdm is not None:
        iterator = tqdm(files, desc="CSV segments", unit="file")

    for csv_path, label in iterator:
        acc, gyro = _load_csv_streams(csv_path)
        if len(acc) < 16:
            continue
        t0_ms = int(acc[0][0])
        t1_ms = int(acc[-1][0])
        duration = (t1_ms - t0_ms) * 1e-3
        if duration < window_s * 0.75:
            continue

        start = 0.0
        while start + window_s <= duration + 1e-6:
            acc_w, gyro_w = window_slice_by_time(acc, gyro, start, start + window_s, t0_ms)
            _, mag_acc, mag_gyro = build_mag_series_from_samples(acc_w, gyro_w, target_fs)
            feat = featurize_mag_series(mag_acc, mag_gyro, target_fs)
            if feat is not None:
                X_list.append(feat)
                y_list.append(label)
            start += stride_s

    if not X_list:
        raise SystemExit(
            "No training windows produced. Try smaller --window-s or more / longer CSVs."
        )
    return np.vstack(X_list), np.array(y_list, dtype=object)


def _fit_centroids(X, y):
    classes = sorted(set(str(v) for v in y))
    rows = []
    for c in classes:
        m = y == c
        rows.append(np.mean(X[m], axis=0))
    return classes, np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(description="Train few-shot activity model from labeled_data/")
    ap.add_argument(
        "--labeled-root",
        default="./labeled_data",
        help="Directory containing session_* folders with manifest.jsonl",
    )
    ap.add_argument("--window-s", type=float, default=2.0, help="Window length in seconds")
    ap.add_argument("--stride-s", type=float, default=1.0, help="Stride between windows")
    ap.add_argument("--target-fs", type=float, default=40.0, help="Resample rate for features")
    ap.add_argument(
        "--out",
        default="./models/activity_fewshot.json",
        help="Output JSON path (centroid few-shot model)",
    )
    ap.add_argument(
        "--sklearn-out",
        default=None,
        metavar="PATH",
        help="If set, also train RandomForest and write this joblib (requires working scikit-learn)",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.labeled_root)
    X, y = collect_windows(
        root,
        window_s=args.window_s,
        stride_s=args.stride_s,
        target_fs=args.target_fs,
    )

    classes, centroids = _fit_centroids(X, y)
    meta = {
        "feature_names": feature_names(),
        "window_s": float(args.window_s),
        "stride_s": float(args.stride_s),
        "target_fs": float(args.target_fs),
        "classes": classes,
        "n_windows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "labeled_root": root,
    }
    bundle_json = {
        "version": 1,
        "kind": "centroid_l2",
        "classes": classes,
        "centroids": centroids.tolist(),
        "meta": meta,
    }

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as jf:
        json.dump(bundle_json, jf, indent=2)
    meta_path = out_path.replace(".json", "_meta.json")
    if meta_path != out_path:
        with open(meta_path, "w", encoding="utf-8") as jf:
            json.dump(meta, jf, indent=2)

    print("Wrote centroid model: {}".format(out_path), flush=True)
    print("Wrote meta:           {}".format(meta_path), flush=True)
    print("Classes: {} | windows: {}".format(classes, meta["n_windows"]), flush=True)

    if args.sklearn_out:
        try:
            import joblib
            from sklearn.ensemble import RandomForestClassifier
        except Exception as e:
            print("Skipping sklearn (--sklearn-out): {}".format(e), flush=True)
            return

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X, y)
        sk_path = os.path.abspath(args.sklearn_out)
        os.makedirs(os.path.dirname(sk_path) or ".", exist_ok=True)
        joblib.dump({"model": clf, "meta": meta}, sk_path)
        print("Wrote sklearn model: {}".format(sk_path), flush=True)


if __name__ == "__main__":
    main()
