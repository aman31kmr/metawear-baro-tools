#!/usr/bin/env python3
"""
Tkinter lab UI: connect to MetaWear, then record labeled IMU segments.

Each segment is one CSV file under ``<out_dir>/session_<timestamp>/`` with a constant ``activity`` column
on every row (plus ``utc_iso``, ``epoch_ms``, ``sensor``, ``x``, ``y``, ``z``). ``manifest.jsonl`` lists
each completed segment with sample counts.

Mounting: defaults to wrist (see ``--mount``); filenames include the mount name for your notes.

Why collect at all?
  Public phone IMU models are rarely a perfect match for a wrist-mounted MetaWear stream (different
  frame rate, noise, and motion patterns). A small amount of your own labeled data (tens of minutes
  total) usually beats a big off-the-shelf model with zero adaptation. If you still want to try a
  pretrained pipeline first, export the same CSV format and fine-tune a tiny classifier on top.

Usage (from repo root):
  python3 har_imu/record_lab_ui.py CF:96:FE:AD:63:E9 --out-dir ./labeled_data
  python3 har_imu/record_lab_ui.py CF:96:FE:AD:63:E9 --hci AA:BB:CC:DD:EE:FF --out-dir ./labeled_data

If BLE reports a connect timeout (``Timed out while trying to connect to remote device``), this UI
retries the full connect a few times before showing an error (same idea as ``connect_device`` in
``metawear_baro_stream.py``).

Turn-detection labeling:
  - standing: just ``standing``
  - walking / running: choose ``straight`` / ``turn_left`` / ``turn_right`` variants.
    CSV rows store the combined label, e.g. ``walking_turn_left``.
    ``manifest.jsonl`` stores both ``activity`` (walking/running/standing) and ``path`` (straight/turn_left/turn_right).
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from threading import Thread
from time import sleep

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as e:
    print("Tkinter is required for this UI:", e, file=sys.stderr)
    sys.exit(1)

from metawear_imu_stream import IMUStreamer


def _log(widget, msg):
    widget.insert(tk.END, msg + "\n")
    widget.see(tk.END)


def main():
    parser = argparse.ArgumentParser(description="Tk UI for labeled IMU recording (wrist by default).")
    parser.add_argument("mac", help="BLE MAC of the board")
    parser.add_argument("--hci", default=None, help="HCI adapter MAC (Linux)")
    parser.add_argument(
        "--out-dir",
        default="labeled_data",
        metavar="DIR",
        help="Root directory for session folders (default: ./labeled_data)",
    )
    parser.add_argument(
        "--mount",
        default="wrist",
        help="Short tag for filenames / manifest (default: wrist)",
    )
    args = parser.parse_args()

    out_root = os.path.abspath(args.out_dir)
    mount = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.mount.strip()) or "wrist"

    streamer = {"obj": None}
    worker_thread = {"t": None}
    running = {"flag": False}
    session_dir = {"path": None}
    recording = {"label": None, "path": None, "counts0": None}
    LABEL_GROUPS = [
        ("standing", [None]),
        ("walking", ["straight", "turn_left", "turn_right"]),
        ("running", ["straight", "turn_left", "turn_right"]),
    ]
    def _label(activity, path):
        return activity if not path else "{}_{}".format(activity, path)

    take_idx = {}
    for act, paths in LABEL_GROUPS:
        for p in paths:
            take_idx[_label(act, p)] = 0

    root = tk.Tk()
    root.title("MetaWear IMU — labeled recording")
    root.geometry("820x640")

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Device MAC").grid(row=0, column=0, sticky="w")
    mac_var = tk.StringVar(value=args.mac)
    ttk.Entry(frm, textvariable=mac_var, width=28).grid(row=0, column=1, sticky="w", padx=(8, 0))

    ttk.Label(frm, text="HCI (optional)").grid(row=1, column=0, sticky="w", pady=(6, 0))
    hci_var = tk.StringVar(value=args.hci or "")
    ttk.Entry(frm, textvariable=hci_var, width=28).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

    ttk.Label(frm, text="Output root").grid(row=2, column=0, sticky="w", pady=(6, 0))
    out_var = tk.StringVar(value=out_root)
    ttk.Entry(frm, textvariable=out_var, width=52).grid(row=2, column=1, sticky="we", padx=(8, 0), pady=(6, 0))

    session_var = tk.StringVar(value="Session: (not started)")
    ttk.Label(frm, textvariable=session_var, foreground="#065f46").grid(
        row=3, column=0, columnspan=2, sticky="w", pady=(10, 0)
    )

    for c in range(2):
        frm.columnconfigure(c, weight=1 if c == 1 else 0)

    log_box = tk.Text(frm, height=14, wrap="word", font=("ui-monospace", 10))
    log_box.grid(row=10, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
    frm.rowconfigure(10, weight=1)

    def ensure_session():
        base = os.path.abspath(out_var.get().strip() or out_root)
        os.makedirs(base, exist_ok=True)
        if session_dir["path"] and os.path.isdir(session_dir["path"]):
            return base
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(base, "session_" + ts)
        os.makedirs(path, exist_ok=True)
        session_dir["path"] = path
        session_var.set("Session: " + path)
        _log(log_box, "Created session folder:\n  " + path)
        return base

    def manifest_append(rec):
        if not session_dir["path"]:
            return
        mp = os.path.join(session_dir["path"], "manifest.jsonl")
        with open(mp, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def worker():
        try:
            hci = hci_var.get().strip() or None
            mac = mac_var.get().strip()
            max_connect_rounds = 5
            st = None
            for boot_i in range(1, max_connect_rounds + 1):
                if not running["flag"]:
                    return
                st = None
                try:
                    st = IMUStreamer(
                        mac,
                        hci=hci,
                        csv_path=None,
                        series_maxlen=0,
                        activity_estimator=None,
                    )
                    streamer["obj"] = st
                    st.start()
                    _log(log_box, "BLE streaming started (no file until you press a Start button).")
                    break
                except BaseException as e:
                    streamer["obj"] = None
                    if st is not None:
                        try:
                            st.stop()
                        except Exception:
                            pass
                    msg_l = str(e).lower()
                    is_timeout = (
                        "timed out" in msg_l
                        and "connect" in msg_l
                        and "remote" in msg_l
                    )
                    if is_timeout and boot_i < max_connect_rounds and running["flag"]:
                        delay = min(2.5, 0.5 * boot_i)
                        _log(
                            log_box,
                            "Connect timed out (remote device); retry {}/{} in {:.1f}s…".format(
                                boot_i,
                                max_connect_rounds,
                                delay,
                            ),
                        )
                        sleep(delay)
                        continue
                    raise
            while running["flag"] and streamer["obj"] is not None:
                sleep(0.08)
        except BaseException as e:
            _log(log_box, "Stream error: " + str(e))
            _log(log_box, traceback.format_exc())
        finally:
            try:
                if streamer["obj"] is not None:
                    streamer["obj"].stop()
            except Exception:
                pass
            streamer["obj"] = None
            running["flag"] = False

            def _done():
                connect_btn.config(state="normal")
                _log(log_box, "Disconnected.")

            root.after(0, _done)

    def on_connect():
        if running["flag"]:
            return
        if not mac_var.get().strip():
            messagebox.showwarning("MAC", "Enter a device MAC.")
            return
        ensure_session()
        running["flag"] = True
        connect_btn.config(state="disabled")
        t = Thread(target=worker, daemon=True)
        worker_thread["t"] = t
        t.start()

    def on_disconnect():
        if recording["label"]:
            messagebox.showinfo("Recording", "Stop the active recording first.")
            return
        running["flag"] = False
        _log(log_box, "Disconnect requested; waiting for stream thread…")

    def start_segment(label):
        if not running["flag"] or streamer["obj"] is None:
            messagebox.showwarning("Connect", "Connect streaming first.")
            return
        if recording["label"]:
            messagebox.showwarning("Busy", 'Already recording "{}". Stop it first.'.format(recording["label"]))
            return
        ensure_session()
        if label not in take_idx:
            take_idx[label] = 0
        take_idx[label] += 1
        fn = "{}_{}_{:04d}.csv".format(label, mount, take_idx[label])
        path = os.path.join(session_dir["path"], fn)
        with streamer["obj"]._lock:
            c0 = dict(streamer["obj"].sample_count)
        streamer["obj"].switch_recording_file(path, activity_label=label)
        recording["label"] = label
        recording["path"] = path
        recording["counts0"] = c0
        _log(log_box, 'Started "{}" -> {}'.format(label, path))

    def stop_segment(label):
        if recording["label"] != label:
            return
        if streamer["obj"] is None:
            return
        path = recording["path"]
        with streamer["obj"]._lock:
            c1 = dict(streamer["obj"].sample_count)
        c0 = recording["counts0"] or {"acc": 0, "gyro": 0, "mag": 0}
        d_acc = c1["acc"] - c0["acc"]
        d_gyro = c1["gyro"] - c0["gyro"]
        d_mag = c1["mag"] - c0["mag"]
        streamer["obj"].switch_recording_file(None)
        # Split combined label into activity + path variant for downstream turn detection.
        parts = (label or "").split("_", 1)
        act = parts[0] if parts else label
        path_kind = parts[1] if len(parts) > 1 else None
        rec = {
            "file": os.path.basename(path),
            "activity": act,
            "path": path_kind,  # straight / turn_left / turn_right (or None for standing)
            "label": label,  # full combined label stored in CSV
            "mount": mount,
            "acc_rows_segment": d_acc,
            "gyro_rows_segment": d_gyro,
            "mag_rows_segment": d_mag,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_append(rec)
        _log(
            log_box,
            'Stopped "{}": acc+{} gyro+{} mag+{} rows (approx.)'.format(label, d_acc, d_gyro, d_mag),
        )
        recording["label"] = None
        recording["path"] = None
        recording["counts0"] = None

    btn_row = 4
    connect_btn = ttk.Button(frm, text="Connect & stream", command=on_connect)
    connect_btn.grid(row=btn_row, column=0, sticky="w", pady=(12, 0))
    ttk.Button(frm, text="Disconnect", command=on_disconnect).grid(
        row=btn_row, column=1, sticky="w", padx=(8, 0), pady=(12, 0)
    )

    def row_for_activity(r, activity, title, paths):
        ttk.Label(frm, text=title, font=("", 11, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(14, 0)
        )
        bf = ttk.Frame(frm)
        bf.grid(row=r + 1, column=0, columnspan=2, sticky="w")
        for p in paths:
            lbl = _label(activity, p)
            pretty = activity if p is None else "{} ({})".format(activity, p)
            ttk.Button(bf, text="Start " + pretty, command=lambda l=lbl: start_segment(l)).pack(side="left")
            ttk.Button(bf, text="Stop " + pretty, command=lambda l=lbl: stop_segment(l)).pack(side="left", padx=(10, 14))

    row = 5
    row_for_activity(row, "standing", "1 — Standing", [None])
    row += 2
    row_for_activity(row, "walking", "2 — Walking", ["straight", "turn_left", "turn_right"])
    row += 2
    row_for_activity(row, "running", "3 — Running", ["straight", "turn_left", "turn_right"])

    ttk.Label(
        frm,
        text=(
            "Tip: keep the watch orientation similar between sessions. "
            "For turn detection, record straight/left/right with consistent turn radius and speed."
        ),
        wraplength=760,
        foreground="#444",
    ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def on_close():
        if recording["label"]:
            if not messagebox.askyesno("Quit", "A recording is active. Stop and quit?"):
                return
            stop_segment(recording["label"])
        running["flag"] = False
        t = worker_thread.get("t")
        if t is not None and t.is_alive():
            t.join(timeout=10.0)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
