#!/usr/bin/env python3
"""
Analyze a long-format IMU CSV from metawear_imu_stream.py.

Handles legacy 5-column files and 6-column (epoch_ms) files, including **mixed**
files where an old header was kept but new rows appended (fixed parser).
"""

from __future__ import print_function

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime

from imu_csv_format import parse_imu_csv_row

try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(iterable, **_kwargs):
        return iterable


def _parse_iso(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Verify IMU CSV counts and time alignment.")
    parser.add_argument(
        "csv",
        nargs="?",
        default="imu.csv",
        metavar="PATH",
        help="Path to IMU CSV (default: ./imu.csv)",
    )
    args = parser.parse_args()

    counts = defaultdict(int)
    epoch_seen = defaultdict(set)
    epochs_per_sensor = {"acc": set(), "gyro": set(), "mag": set()}
    utc_per_sensor = {"acc": [], "gyro": [], "mag": []}
    epoch_list_per_sensor = {"acc": [], "gyro": [], "mag": []}
    pair_counts = defaultdict(int)
    fmt_legacy = 0
    fmt_with_epoch = 0
    skipped = 0

    with open(args.csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print("Empty file.", file=sys.stderr)
            sys.exit(1)
        has_epoch_header = any(h.strip().lower() == "epoch_ms" for h in header)
        print("--- CSV header ---")
        print("  Columns: {}".format(",".join(header)))
        print("  Header lists epoch_ms: {}".format(has_epoch_header))

        for parts in tqdm(reader, desc="Read rows", unit="row"):
            if not parts or (parts[0].strip().lower().startswith("utc_iso")):
                skipped += 1
                continue
            rec = parse_imu_csv_row(parts)
            if rec is None:
                skipped += 1
                continue
            if rec["epoch_ms"] is None:
                fmt_legacy += 1
            else:
                fmt_with_epoch += 1
            sensor = rec["sensor"]
            counts[sensor] += 1
            ts = rec.get("utc_iso")
            if ts:
                t = _parse_iso(ts)
                if t is not None:
                    utc_per_sensor[sensor].append(t)
            epoch_ms = rec.get("epoch_ms")
            if epoch_ms is not None:
                pair_counts[(sensor, epoch_ms)] += 1
                epoch_seen[epoch_ms].add(sensor)
                epochs_per_sensor[sensor].add(epoch_ms)
                epoch_list_per_sensor[sensor].append(epoch_ms)

    print("\n--- Row format (detected from content) ---")
    print("  Rows with board epoch_ms: {}".format(fmt_with_epoch))
    print("  Legacy rows (host time only): {}".format(fmt_legacy))
    print("  Skipped / non-data lines: {}".format(skipped))
    if fmt_with_epoch and not has_epoch_header:
        print(
            "\n  NOTE: File has legacy header but newer rows include epoch_ms in column 2.",
            "\n        Older tools that only read the header mis-counted. Parser is corrected here.",
        )

    n_total = sum(counts.values())
    dup_rows = defaultdict(int)
    for (sns, _), c in pair_counts.items():
        if c > 1:
            dup_rows[sns] += c - 1

    print("\n--- Per-sensor rows (fusion = three streams; counts often differ slightly) ---")
    for key in ("acc", "gyro", "mag"):
        print(
            "  {:4s}: {:6d}".format(key, counts.get(key, 0)),
            "  duplicate (sensor, epoch_ms): {}".format(dup_rows.get(key, 0)),
        )
    print("  TOTAL: {}".format(n_total))

    epochs_with_bundle = sorted(
        e for e, st in epoch_seen.items() if st >= {"acc", "gyro", "mag"}
    )
    epochs_any = sorted(epoch_seen.keys())

    print("\n--- Board epoch_ms (fusion clock) ---")
    if not epochs_any:
        print("  No epoch data in any row — use host utc_iso only.")
    else:
        print("  Unique epoch_ms ticks (all sensors): {}".format(len(epochs_any)))
        for key in ("acc", "gyro", "mag"):
            print("  Distinct epochs with {:4s}: {:6d}".format(key, len(epochs_per_sensor[key])))

        triple = len(epochs_with_bundle)
        print("  Epochs where acc+gyro+mag **all** appear: {}".format(triple))
        if triple and len(epochs_any):
            pct = 100.0 * triple / len(epochs_any)
            print("    ({:.1f}% of distinct epoch ticks)".format(pct))

        print("\n--- Epoch spacing (|Δepoch| ms, CSV order per sensor) ---")
        for key in ("acc", "gyro", "mag"):
            lst = epoch_list_per_sensor[key]
            if len(lst) < 2:
                print("  {:4s}: n/a".format(key))
                continue
            deltas = [abs(lst[i] - lst[i - 1]) for i in range(1, len(lst))]
            deltas.sort()
            med = deltas[len(deltas) // 2]
            print(
                "  {:4s}: median |Δepoch| {} ms ({} intervals)".format(key, med, len(deltas)),
            )

    print("\n--- Host utc_iso (callback time; not cross-sensor sync) ---")
    for key in ("acc", "gyro", "mag"):
        utc = utc_per_sensor[key]
        if len(utc) < 2:
            continue
        dhost = [(utc[i] - utc[i - 1]).total_seconds() * 1000.0 for i in range(1, len(utc))]
        dhost.sort()
        med = dhost[len(dhost) // 2]
        print("  {:4s}: median |Δutc| {:.2f} ms".format(key, med))

    max_n = max(counts.get("acc", 0), counts.get("gyro", 0), counts.get("mag", 0), 1)
    min_n = min(counts.get("acc", 0), counts.get("gyro", 0), counts.get("mag", 0))
    print("\n--- Count spread ---")
    print("  max(n) − min(n) = {}".format(max_n - min_n))
    print(
        "\nInterpretation:\n"
        "  Different n per stream is normal (separate callbacks).\n"
        "  Align samples with epoch_ms when present; utc_iso is receive-time only.\n",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
