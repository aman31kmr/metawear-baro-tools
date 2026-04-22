#!/usr/bin/env python3
"""
Stream Bosch barometer pressure (Pa) from a MetaMotion / MetaWear board via official MetaWear Python SDK.

Jetson / Linux checklist:
  - Bluetooth enabled: `sudo rfkill unblock bluetooth` if soft-blocked
  - BlueZ running: `systemctl status bluetooth`
  - Permissions: user in `bluetooth` group, or run with `sudo` for scanning
  - Disconnect the board from phones/tablets (MetaBase) so BLE is free
  - If multiple adapters: pass `--hci` with `hciconfig` / `bluetoothctl list` MAC

Usage:
  python3 metawear_baro_stream.py --scan --scan-seconds 8
  python3 metawear_baro_stream.py CF:96:FE:AD:63:E9 --samples 200
  python3 metawear_baro_stream.py CF:96:FE:AD:63:E9 --altitude --seconds 30 --csv ./stairs.csv
  With --csv, a matching .png (e.g. stairs.png) is written at the end of the run.
"""
from __future__ import print_function

import argparse
import csv
import sys
from datetime import datetime, timezone
from threading import Event
from time import monotonic, sleep
from queue import Queue, Empty
from threading import Thread
from collections import deque
from math import pow
import traceback
import json
from threading import Lock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import webbrowser
import signal
import os

_INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MetaWear Height Live</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; color: #111827; }
    .row { display: flex; gap: 18px; align-items: baseline; flex-wrap: wrap; }
    .pill { background: #f3f4f6; padding: 6px 10px; border-radius: 999px; font-size: 14px; }
    canvas { width: 100%; height: 420px; border: 1px solid #d1d5db; border-radius: 10px; }
    .muted { color: #6b7280; font-size: 13px; }
    a { color: #2563eb; }
    button { background: #ef4444; border: 0; color: white; padding: 8px 12px; border-radius: 10px; cursor: pointer; font-weight: 600; }
    button:disabled { background: #fca5a5; cursor: not-allowed; }
  </style>
</head>
<body>
  <h2 style="margin: 0 0 10px 0;">MetaWear Height (live)</h2>
  <div class="row" style="margin-bottom: 10px;">
    <div class="pill">Samples: <span id="n">0</span></div>
    <div class="pill">Latest: <span id="latest">—</span></div>
    <div class="pill">Window: <span id="win">all</span></div>
    <button id="stopBtn" title="Stops streaming and saves CSV + PNG">Stop</button>
    <div class="muted">Streaming continues until you press Stop.</div>
  </div>
  <canvas id="chart" width="1200" height="420"></canvas>
  <div class="muted" style="margin-top: 8px;">If the line is flat, you’re stationary; climb/descend to see motion.</div>

<script>
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
const elN = document.getElementById('n');
const elLatest = document.getElementById('latest');
const elWin = document.getElementById('win');
const stopBtn = document.getElementById('stopBtn');

// windowS = 0 means "show the whole run so far"
let windowS = 0;
elWin.textContent = 'all';

function drawSeries(t, y) {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  // margins
  const ml=60, mr=16, mt=16, mb=34;
  const iw = Math.max(10, w-ml-mr), ih = Math.max(10, h-mt-mb);

  // axes
  ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(ml, h-mb); ctx.lineTo(w-mr, h-mb);
  ctx.moveTo(ml, mt); ctx.lineTo(ml, h-mb);
  ctx.stroke();

  if (!t.length) {
    ctx.fillStyle = '#6b7280';
    ctx.fillText('waiting for samples...', ml+8, mt+18);
    return;
  }

  const t0 = t[0], t1 = t[t.length-1] > t0 ? t[t.length-1] : t0 + 1.0;
  let y0 = Math.min(...y), y1 = Math.max(...y);
  if (y1 === y0) { y0 -= 0.5; y1 += 0.5; }
  else { const pad = Math.max(0.05, 0.15*(y1-y0)); y0 -= pad; y1 += pad; }

  const sx = (x)=> ml + (x - t0) * iw / (t1 - t0);
  const sy = (v)=> mt + (y1 - v) * ih / (y1 - y0);

  // grid
  ctx.strokeStyle = '#eef2f7';
  for (const frac of [0.25,0.5,0.75]) {
    const yy = mt + frac*ih;
    ctx.beginPath(); ctx.moveTo(ml, yy); ctx.lineTo(w-mr, yy); ctx.stroke();
  }

  // labels
  ctx.fillStyle = '#6b7280'; ctx.font = '12px system-ui';
  ctx.fillText(`${t0.toFixed(0)}s`, ml, h-10);
  ctx.textAlign = 'right';
  ctx.fillText(`${t1.toFixed(0)}s`, w-mr, h-10);
  ctx.textAlign = 'left';
  ctx.fillText(`${y1.toFixed(2)}m`, 6, mt+12);
  ctx.fillText(`${y0.toFixed(2)}m`, 6, h-mb);

  // series
  ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i=0;i<t.length;i++){
    const xx=sx(t[i]), yy=sy(y[i]);
    if (i===0) ctx.moveTo(xx,yy); else ctx.lineTo(xx,yy);
  }
  ctx.stroke();

  // last point marker
  const lx = sx(t[t.length-1]), ly = sy(y[y.length-1]);
  ctx.fillStyle = '#2563eb';
  ctx.beginPath(); ctx.arc(lx, ly, 3.5, 0, 2*Math.PI); ctx.fill();
}

async function tick(){
  try{
    const r = await fetch(`/delta?from=${nextFrom}`, {cache:'no-store'});
    const j = await r.json();
    elN.textContent = j.n ?? 0;
    if (j.err){
      // stream stopped or server error
      stopBtn.disabled = true;
      stopBtn.textContent = 'Stopped';
      return;
    }

    // append new points and keep a bounded client-side window for render speed
    if (j.t && j.y && j.t.length){
      for (let i=0;i<j.t.length;i++){
        tAll.push(j.t[i]);
        yAll.push(j.y[i]);
      }
      const last = yAll[yAll.length-1];
      elLatest.textContent = `${last.toFixed(3)} m`;
      nextFrom = j.next_from ?? nextFrom;
    }

    // render: last N seconds (or last ~5000 points if windowS==0)
    if (!tAll.length){
      drawSeries([], []);
    } else if (windowS > 0){
      const tLast = tAll[tAll.length-1];
      const tMin = Math.max(0, tLast - windowS);
      let k = 0;
      while (k < tAll.length && tAll[k] < tMin) k++;
      drawSeries(tAll.slice(k), yAll.slice(k));
    } else {
      // whole run so far
      drawSeries(tAll, yAll);
      const spanS = Math.max(0, tAll[tAll.length-1] - tAll[0]);
      elWin.textContent = `${spanS.toFixed(0)}s`;
    }
  }catch(e){
    // ignore transient errors
  }
  setTimeout(tick, 120);
}

let nextFrom = 0;
const tAll = [];
const yAll = [];
tick();

stopBtn.addEventListener('click', async ()=>{
  stopBtn.disabled = true;
  stopBtn.textContent = 'Stopping...';
  try{
    await fetch('/stop', {method:'POST'});
    stopBtn.textContent = 'Stopped';
  }catch(e){
    stopBtn.textContent = 'Stop failed';
  }
});
</script>
</body>
</html>
"""

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, **_kwargs):
        if iterable is None:
            class _Noop:
                n = 0

                def update(self, n=1):
                    self.n += n

                def set_postfix(self, **_kw):
                    pass

                def close(self):
                    pass

            return _Noop()
        return iterable

from mbientlab.metawear import MetaWear, libmetawear, parse_value

from metawear_connect_safe import metawear_connect

from mbientlab.metawear.cbindings import (
    BaroBoschOversampling,
    FnVoid_VoidP_DataP,
)


def connect_device(device):
    """
    Complete connect + SDK init, then let the BLE link settle before sensor GATT traffic.

    MetaWear Python docs recommend `mbl_mw_settings_set_connection_parameters` and a short
    pause before configuring streams; without this, Warble often reports writes
    \"Error trying to issue command mid state\" when config/subscribe/start overlap
    an in-flight GATT operation.
    """
    print("Connecting (BLE + board init) ...", flush=True)
    # Warble/BlueZ can temporarily report "Operation now in progress" or "resource busy"
    # after previous crashes / stale connections. Retry a few times before giving up.
    last_err = None
    for attempt in range(1, 6):
        try:
            metawear_connect(device)
            last_err = None
            break
        except BaseException as err:
            last_err = err
            msg = str(err)
            retryable = ("Device or resource busy" in msg) or ("Operation now in progress" in msg)
            if attempt < 6 and retryable:
                backoff = min(2.0, 0.25 * attempt)
                print(
                    f"Connect attempt {attempt} failed ({msg}); retrying in {backoff:.2f}s ...",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    device.disconnect()
                except BaseException:
                    pass
                sleep(backoff)
                continue
            print("Connect failed: {}".format(err), file=sys.stderr)
            raise
    if last_err is not None:
        raise last_err
    if not device.is_connected:
        raise RuntimeError("connect() returned but device.is_connected is False")
    if device.in_metaboot_mode:
        raise RuntimeError(
            "Board is in MetaBoot (DFU) mode; exit DFU or flash firmware, then retry."
        )
    # Same pattern as mbientlab streaming examples: tune connection interval, then wait
    # for the stack to finish so subsequent command writes do not race Warble state.
    libmetawear.mbl_mw_settings_set_connection_parameters(device.board, 7.5, 7.5, 0, 6000)
    sleep(1.5)
    print("Connected and ready for sensors:", device.address, flush=True)


def run_scan(hci, scan_seconds):
    from mbientlab.warble import BleScanner

    seen = {}

    def on_device(result):
        mac = result.mac
        if mac not in seen:
            seen[mac] = (result.name, result.rssi)
            print("{}\t{}\t{}".format(mac, result.name or "?", result.rssi))

    BleScanner.set_handler(on_device)
    if hci:
        BleScanner.start(hci=hci)
    else:
        BleScanner.start()
    try:
        sleep(scan_seconds)
    finally:
        BleScanner.stop()
    print("Scan finished; {} unique device(s) seen.".format(len(seen)), file=sys.stderr)


def save_csv_plot_session(csv_path, times_utc, values, altitude):
    """
    Save a line plot for the samples collected in this run (same data as written to CSV).
    Output path: same as csv_path with extension replaced by .png
    """
    import os

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if not times_utc:
        return
    out_png = os.path.splitext(csv_path)[0] + ".png"
    t0 = times_utc[0]
    x_sec = []
    for t in tqdm(times_utc, desc="Plot series", unit="pt", leave=False):
        x_sec.append((t - t0).total_seconds())

    y_label = "Altitude (m, barometer-derived)" if altitude else "Pressure (Pa)"
    title = os.path.basename(csv_path)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.plot(x_sec, values, color="#2563eb", linewidth=1.2, marker="o", markersize=3, alpha=0.85)
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Saved plot: {}".format(out_png), flush=True)


class BaroStreamer:
    def __init__(
        self,
        mac,
        *,
        hci=None,
        altitude=False,
        csv_path="baro_stream.csv",
        ui_height_from_pressure=False,
        series_maxlen=0,
    ):
        self.mac = mac
        self.hci = hci
        self.altitude = altitude
        self.csv_path = csv_path
        self.ui_height_from_pressure = ui_height_from_pressure

        self.device = None
        self.board = None
        self.sig = None

        self.done = Event()
        self.callbacks = []

        self.csv_file = None
        self.csv_writer = None
        self.plot_times_utc = []
        self.plot_values = []

        self.sample_count = 0
        self.latest_value = None
        self.latest_ts_utc = None
        self._p0_pa = None
        self._lock = Lock()
        # (t_sec, height_m). If series_maxlen <= 0, keep all points until stop.
        self._series_maxlen = int(series_maxlen)
        self._series = [] if self._series_maxlen <= 0 else deque(maxlen=self._series_maxlen)
        self._t0_utc = None

        self._ui_queue = None

    @staticmethod
    def _height_from_pressure_m(p_pa, p0_pa):
        # International Standard Atmosphere approximation (good for relative changes).
        # h = 44330 * (1 - (P/P0)^0.1903)
        return 44330.0 * (1.0 - pow(float(p_pa) / float(p0_pa), 0.1903))

    def _open_csv(self):
        if not self.csv_path:
            return
        import os

        new_file = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        self.csv_file = open(self.csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if new_file:
            if self.altitude or self.ui_height_from_pressure:
                self.csv_writer.writerow(["utc_iso", "altitude_m"])
            else:
                self.csv_writer.writerow(["utc_iso", "pressure_pa"])

    def start(self, *, ui_queue=None):
        self._ui_queue = ui_queue
        kwargs = {}
        if self.hci:
            kwargs["hci_mac"] = self.hci
        self.device = MetaWear(self.mac, **kwargs)
        connect_device(self.device)
        self.board = self.device.board

        libmetawear.mbl_mw_baro_bosch_set_oversampling(self.board, BaroBoschOversampling.LOW_POWER)
        libmetawear.mbl_mw_baro_bosch_set_standby_time(self.board, 500.0)
        libmetawear.mbl_mw_baro_bosch_write_config(self.board)

        def data_handler(_ctx, data):
            val = parse_value(data)
            ts = datetime.now(timezone.utc)
            out_val = float(val)
            if self.ui_height_from_pressure:
                # Here val is pressure (Pa). Convert to relative height (m).
                p_pa = float(val)
                if self._p0_pa is None:
                    self._p0_pa = p_pa
                    out_val = 0.0
                else:
                    out_val = self._height_from_pressure_m(p_pa, self._p0_pa)

            self.latest_value = float(out_val)
            self.latest_ts_utc = ts
            self.sample_count += 1

            if self.csv_writer is not None:
                self.csv_writer.writerow([ts.isoformat(), "{:.6f}".format(out_val)])
                self.csv_file.flush()
                self.plot_times_utc.append(ts)
                self.plot_values.append(float(out_val))

            if self._ui_queue is not None:
                # Keep UI lightweight: only send newest sample and count.
                self._ui_queue.put((ts, float(out_val), self.sample_count))

            # Always retain series for web UI / plotting
            with self._lock:
                if self._t0_utc is None:
                    self._t0_utc = ts
                t_sec = (ts - self._t0_utc).total_seconds()
                if isinstance(self._series, list):
                    self._series.append((t_sec, float(out_val)))
                else:
                    self._series.append((t_sec, float(out_val)))

        cb = FnVoid_VoidP_DataP(data_handler)
        self.callbacks.append(cb)

        # For UI live plotting we always subscribe to pressure and compute height locally
        # (more robust than the SDK altitude signal on some stacks).
        if self.ui_height_from_pressure:
            self.sig = libmetawear.mbl_mw_baro_bosch_get_pressure_data_signal(self.board)
        else:
            if self.altitude:
                self.sig = libmetawear.mbl_mw_baro_bosch_get_altitude_data_signal(self.board)
            else:
                self.sig = libmetawear.mbl_mw_baro_bosch_get_pressure_data_signal(self.board)

        self._open_csv()
        libmetawear.mbl_mw_datasignal_subscribe(self.sig, None, cb)
        libmetawear.mbl_mw_baro_bosch_start(self.board)

    def stop(self):
        if self.done.is_set():
            return
        self.done.set()
        try:
            if self.sig is not None:
                libmetawear.mbl_mw_datasignal_unsubscribe(self.sig)
            if self.board is not None:
                libmetawear.mbl_mw_baro_bosch_stop(self.board)
        finally:
            try:
                if self.csv_file is not None:
                    self.csv_file.close()
                    if self.plot_times_utc:
                        save_csv_plot_session(
                            self.csv_path, self.plot_times_utc, self.plot_values, self.altitude
                        )
            finally:
                if self.device is not None:
                    self.device.disconnect()

    def snapshot_series(self, *, window_s=0.0):
        with self._lock:
            if not self._series:
                return {"t": [], "y": [], "n": self.sample_count, "next_from": 0}
            if window_s is None or float(window_s) <= 0.0:
                t = [tt for tt, _ in self._series]
                y = [yy for _, yy in self._series]
                return {"t": t, "y": y, "n": self.sample_count, "next_from": len(t)}
            t_last = self._series[-1][0]
            t_min = max(0.0, t_last - float(window_s))
            t = []
            y = []
            for tt, yy in self._series:
                if tt >= t_min:
                    t.append(tt)
                    y.append(yy)
            return {"t": t, "y": y, "n": self.sample_count, "next_from": len(self._series)}

    def snapshot_delta(self, *, from_idx=0):
        """
        Return only new points since from_idx.
        from_idx is interpreted as an index into the full in-memory series (0-based).
        """
        try:
            from_i = int(from_idx)
        except Exception:
            from_i = 0
        if from_i < 0:
            from_i = 0
        with self._lock:
            n_total = len(self._series)
            if from_i >= n_total:
                return {"t": [], "y": [], "n": self.sample_count, "next_from": n_total}
            chunk = list(self._series[from_i:n_total]) if isinstance(self._series, list) else list(self._series)[from_i:n_total]
            t = [tt for tt, _ in chunk]
            y = [yy for _, yy in chunk]
            return {"t": t, "y": y, "n": self.sample_count, "next_from": n_total}


def run_ui(mac, *, hci=None, altitude=False, csv_path="baro_stream.csv"):
    import tkinter as tk
    from tkinter import ttk

    q = Queue()
    streamer = BaroStreamer(
        mac,
        hci=hci,
        altitude=altitude,
        csv_path=csv_path,
        ui_height_from_pressure=True,
    )

    root = tk.Tk()
    root.title("MetaWear Height Stream (Live)")
    root.geometry("900x520")

    status_var = tk.StringVar(value="Idle")
    value_var = tk.StringVar(value="—")
    count_var = tk.StringVar(value="0")
    elapsed_var = tk.StringVar(value="0.0 s")

    t_start = {"mono": None}
    running = {"flag": False}
    worker = {"thread": None}

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    hdr = ttk.Label(frm, text=f"Device: {mac}    Live height (m) from barometer")
    hdr.pack(anchor="w")

    grid = ttk.Frame(frm)
    grid.pack(fill="x", pady=(12, 8))

    ttk.Label(grid, text="Status").grid(row=0, column=0, sticky="w")
    ttk.Label(grid, textvariable=status_var).grid(row=0, column=1, sticky="w", padx=(10, 0))

    ttk.Label(grid, text="Latest height").grid(row=1, column=0, sticky="w")
    ttk.Label(grid, textvariable=value_var).grid(row=1, column=1, sticky="w", padx=(10, 0))

    ttk.Label(grid, text="Samples").grid(row=2, column=0, sticky="w")
    ttk.Label(grid, textvariable=count_var).grid(row=2, column=1, sticky="w", padx=(10, 0))

    ttk.Label(grid, text="Elapsed").grid(row=3, column=0, sticky="w")
    ttk.Label(grid, textvariable=elapsed_var).grid(row=3, column=1, sticky="w", padx=(10, 0))

    for i in range(2):
        grid.columnconfigure(i, weight=1)

    # Live plot (pure Tk canvas; avoids Matplotlib/Tk backend issues on embedded systems)
    plot_wrap = ttk.Frame(frm)
    plot_wrap.pack(fill="both", expand=True, pady=(6, 6))

    plot = tk.Canvas(plot_wrap, bg="white", highlightthickness=1, highlightbackground="#d1d5db")
    plot.pack(fill="both", expand=True)

    plot_title = ttk.Label(plot_wrap, text="Height (relative) - live", foreground="#111827")
    plot_title.place(x=10, y=6)

    btns = ttk.Frame(frm)
    btns.pack(fill="x", pady=(6, 0))

    def set_ui_running(is_running):
        running["flag"] = is_running
        start_btn["state"] = "disabled" if is_running else "normal"
        stop_btn["state"] = "normal" if is_running else "disabled"

    def worker_start():
        try:
            streamer.start(ui_queue=q)
            q.put(("__status__", "Streaming"))
            # Keep the worker alive until stop() is called.
            while not streamer.done.is_set():
                sleep(0.05)
        except BaseException as e:
            q.put(("__error__", "{}\n{}".format(e, traceback.format_exc())))
        finally:
            q.put(("__stopped__", None))

    def on_start():
        if running["flag"]:
            return
        status_var.set("Connecting…")
        set_ui_running(True)
        t_start["mono"] = monotonic()
        worker["thread"] = Thread(target=worker_start, daemon=True)
        worker["thread"].start()

    def on_stop():
        if not running["flag"]:
            root.destroy()
            return
        status_var.set("Stopping… (will save + plot)")
        try:
            streamer.stop()
        except BaseException as e:
            status_var.set(f"Stop error: {e}")

    start_btn = ttk.Button(btns, text="Start streaming", command=on_start)
    stop_btn = ttk.Button(btns, text="Stop + save + plot", command=on_stop, state="disabled")
    start_btn.pack(side="left")
    stop_btn.pack(side="left", padx=(10, 0))

    foot = ttk.Label(
        frm,
        text=f"CSV: {csv_path}    (plot written next to CSV on stop)",
        foreground="#555",
    )
    foot.pack(anchor="w", pady=(12, 0))

    # Keep the last N seconds in the live chart.
    window_s = 30.0
    t0 = {"utc": None, "mono": None}
    xs = deque()
    ys = deque()
    last_draw = {"mono": 0.0}

    def _draw_plot():
        # Throttle to keep UI smooth.
        now = monotonic()
        if now - last_draw["mono"] < 0.10:
            return
        last_draw["mono"] = now

        w = plot.winfo_width()
        h = plot.winfo_height()
        if w <= 2 or h <= 2:
            return

        # Margins for axes labels
        ml, mr, mt, mb = 55, 15, 28, 28
        iw = max(10, w - ml - mr)
        ih = max(10, h - mt - mb)

        plot.delete("all")
        # Border / title handled by frame/label; just axes + series here.

        if not xs:
            # Axes placeholder
            plot.create_line(ml, h - mb, w - mr, h - mb, fill="#9ca3af")  # x-axis
            plot.create_line(ml, mt, ml, h - mb, fill="#9ca3af")  # y-axis
            plot.create_text(
                ml + 6,
                mt + 6,
                anchor="nw",
                text="waiting for samples...",
                fill="#6b7280",
            )
            return

        x0 = xs[0]
        x1 = xs[-1] if len(xs) > 1 else xs[0] + 1.0
        if x1 <= x0:
            x1 = x0 + 1.0

        y0 = min(ys)
        y1 = max(ys)
        if y1 == y0:
            y0 -= 0.5
            y1 += 0.5
        else:
            pad = max(0.05, 0.15 * (y1 - y0))
            y0 -= pad
            y1 += pad

        def sx(x):
            return ml + (float(x) - x0) * iw / (x1 - x0)

        def sy(y):
            # y increases upward
            return mt + (y1 - float(y)) * ih / (y1 - y0)

        # Axes
        plot.create_line(ml, h - mb, w - mr, h - mb, fill="#9ca3af")
        plot.create_line(ml, mt, ml, h - mb, fill="#9ca3af")

        # Simple ticks/labels
        plot.create_text(ml, h - 6, anchor="sw", text=f"{x0:.0f}s", fill="#6b7280")
        plot.create_text(w - mr, h - 6, anchor="se", text=f"{x1:.0f}s", fill="#6b7280")
        plot.create_text(ml - 6, mt, anchor="ne", text=f"{y1:.2f}m", fill="#6b7280")
        plot.create_text(ml - 6, h - mb, anchor="se", text=f"{y0:.2f}m", fill="#6b7280")

        # Grid (light)
        for frac in (0.25, 0.5, 0.75):
            yy = mt + frac * ih
            plot.create_line(ml, yy, w - mr, yy, fill="#eef2f7")

        # Series polyline
        pts = []
        for x, y in zip(xs, ys):
            pts.extend([sx(x), sy(y)])
        if len(pts) >= 4:
            plot.create_line(*pts, fill="#2563eb", width=2)
            # last point marker
            plot.create_oval(pts[-2] - 3, pts[-1] - 3, pts[-2] + 3, pts[-1] + 3, fill="#2563eb", outline="")

    def pump_queue():
        if t_start["mono"] is not None and running["flag"]:
            elapsed_var.set("{:.1f} s".format(monotonic() - t_start["mono"]))

        try:
            # Drain a bounded number of messages per tick so the UI stays responsive
            # even if BLE callbacks enqueue faster than the GUI loop.
            latest_sample = None
            drained = 0
            while drained < 250:
                msg = q.get_nowait()
                drained += 1
                if isinstance(msg, tuple) and msg and msg[0] == "__status__":
                    status_var.set(msg[1])
                elif isinstance(msg, tuple) and msg and msg[0] == "__error__":
                    status_var.set("Error")
                    set_ui_running(False)
                    value_var.set("—")
                    # Print full traceback to terminal so debugging is possible.
                    print("\nUI worker error:\n{}".format(msg[1]), file=sys.stderr, flush=True)
                    root.after(250, root.destroy)
                elif isinstance(msg, tuple) and msg and msg[0] == "__stopped__":
                    status_var.set("Stopped (saved + plotted)")
                    set_ui_running(False)
                else:
                    # Expected: (datetime ts_utc, float val, int n)
                    if isinstance(msg, tuple) and len(msg) == 3:
                        latest_sample = msg
            if latest_sample is not None:
                ts, val, n = latest_sample
                if t0["utc"] is None:
                    t0["utc"] = ts
                    t0["mono"] = monotonic()
                count_var.set(str(n))
                value_var.set("{:.3f} m @ {}".format(val, ts.strftime("%H:%M:%S")))

                # Update live plot series
                x = (ts - t0["utc"]).total_seconds()
                xs.append(x)
                ys.append(val)
                # Drop old points outside the window
                while xs and (xs[-1] - xs[0]) > window_s:
                    xs.popleft()
                    ys.popleft()
                _draw_plot()
        except Empty:
            pass
        except BaseException as e:
            # Surface errors in terminal; UI will close.
            print("UI error:", e, file=sys.stderr)
        # Even if no new sample arrived, periodically redraw (resizes, etc.)
        _draw_plot()
        root.after(50, pump_queue)

    def on_close():
        try:
            streamer.stop()
        except BaseException:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    # Redraw on resize/layout changes
    plot.bind("<Configure>", lambda _e: _draw_plot())
    root.after(50, pump_queue)
    # Ensure an initial draw after geometry is realized
    root.after(250, _draw_plot)
    # Auto-start streaming so you immediately see the live chart.
    root.after(150, on_start)
    root.mainloop()

def run_webui(mac, *, hci=None, csv_path="height.csv", port=8000, window_s=0.0, open_browser=True):
    """
    Live plot in a browser (robust on Jetson): start a local HTTP server and poll JSON.
    Stop with Ctrl+C in the terminal; on stop it saves CSV + png (same as usual).
    """
    streamer = BaroStreamer(
        mac,
        hci=hci,
        altitude=False,
        csv_path=csv_path,
        ui_height_from_pressure=True,
        series_maxlen=0,
    )

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, content_type="text/html; charset=utf-8"):
            body_bytes = body if isinstance(body, (bytes, bytearray)) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def log_message(self, format, *args):
            # keep terminal clean
            return

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index.html"):
                self._send(200, _INDEX_HTML)
                return
            if self.path.startswith("/latest"):
                snap = streamer.snapshot_series(window_s=window_s)
                self._send(200, json.dumps(snap), content_type="application/json")
                return
            if self.path.startswith("/delta"):
                # /delta?from=123
                try:
                    q = self.path.split("?", 1)[1] if "?" in self.path else ""
                    params = {}
                    for kv in q.split("&"):
                        if not kv:
                            continue
                        k, v = kv.split("=", 1) if "=" in kv else (kv, "")
                        params[k] = v
                    from_i = params.get("from", "0")
                except Exception:
                    from_i = "0"
                snap = streamer.snapshot_delta(from_idx=from_i)
                self._send(200, json.dumps(snap), content_type="application/json")
                return
            self._send(404, "not found", content_type="text/plain")

        def do_POST(self):
            if self.path.startswith("/stop"):
                # Acknowledge immediately so the browser doesn't hang.
                self._send(200, "ok", content_type="text/plain")
                # Stop streaming + shut down server in a separate thread
                def _shutdown():
                    try:
                        streamer.stop()
                    except Exception:
                        pass
                    try:
                        httpd.shutdown()
                    except Exception:
                        pass

                Thread(target=_shutdown, daemon=True).start()
                return
            self._send(404, "not found", content_type="text/plain")

    def _pick_host():
        return "127.0.0.1"

    host = _pick_host()
    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    url = f"http://{host}:{int(port)}/"
    print(f"Web UI: {url}", flush=True)
    if open_browser:
        try:
            webbrowser.open(url, new=1, autoraise=True)
        except Exception:
            pass

    # Start streaming (BLE) in a background thread so the HTTP server stays responsive.
    def worker():
        try:
            streamer.start(ui_queue=None)
        except BaseException as e:
            print("Stream start error:", e, file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            try:
                httpd.shutdown()
            except Exception:
                pass

    t = Thread(target=worker, daemon=True)
    t.start()

    stopping = {"flag": False}

    def _request_stop(_signum=None, _frame=None):
        if stopping["flag"]:
            return
        stopping["flag"] = True
        print("\nStopping... (will save CSV + PNG)", flush=True)
        try:
            # serve_forever blocks; shutdown from signal handler via a thread.
            Thread(target=httpd.shutdown, daemon=True).start()
        except Exception:
            pass

    old_int = None
    old_term = None
    try:
        old_int = signal.signal(signal.SIGINT, _request_stop)
        old_term = signal.signal(signal.SIGTERM, _request_stop)
    except Exception:
        old_int = None
        old_term = None

    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        _request_stop()
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass
        try:
            streamer.stop()
        finally:
            # Restore handlers (best effort)
            try:
                if old_int is not None:
                    signal.signal(signal.SIGINT, old_int)
                if old_term is not None:
                    signal.signal(signal.SIGTERM, old_term)
            except Exception:
                pass
        # Make it unambiguous where files were written.
        try:
            base = os.path.splitext(csv_path)[0]
            print(f"Saved CSV: {os.path.abspath(csv_path)}", flush=True)
            print(f"Saved plot: {os.path.abspath(base + '.png')}", flush=True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Stream barometer pressure (Pa) or altitude (m) from MetaWear"
    )
    parser.add_argument("mac", nargs="?", help="BLE MAC of the board (e.g. CF:96:FE:AD:63:E9)")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Samples to collect then exit (default: 100; ignored if --seconds is set)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="Run for this many seconds (overrides --samples), e.g. stair climb logging",
    )
    parser.add_argument(
        "--altitude",
        action="store_true",
        help="Use barometer-derived altitude in meters (SDK) instead of raw pressure (Pa)",
    )
    parser.add_argument("--hci", default=None, help="HCI adapter BLE MAC (Linux), e.g. from hciconfig")
    parser.add_argument("--scan", action="store_true", help="Scan for BLE devices and exit")
    parser.add_argument("--scan-seconds", type=float, default=8.0, help="With --scan, how long to listen")
    parser.add_argument(
        "--csv",
        default="baro_stream.csv",
        metavar="PATH",
        help="Append UTC timestamp + value per row (pressure Pa or altitude m; header if new file)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Open a simple UI for continuous streaming; Stop/close saves CSV and writes plot",
    )
    parser.add_argument(
        "--webui",
        action="store_true",
        help="Open a browser-based live plot (recommended if Tk UI doesn't render)",
    )
    parser.add_argument("--webui-port", type=int, default=8000, help="Port for --webui (default: 8000)")
    parser.add_argument(
        "--webui-window",
        type=float,
        default=0.0,
        help="Seconds shown in the live web chart (0 = whole run; default: 0)",
    )
    args = parser.parse_args()

    if args.seconds is not None:
        if args.seconds <= 0:
            print("--seconds must be > 0", file=sys.stderr)
            sys.exit(2)
    elif args.samples < 1:
        print("--samples must be >= 1", file=sys.stderr)
        sys.exit(2)

    kwargs = {}
    if args.hci:
        kwargs["hci_mac"] = args.hci

    if args.scan:
        run_scan(args.hci, args.scan_seconds)
        return

    if not args.mac:
        print("Provide board MAC or use --scan to discover devices.", file=sys.stderr)
        sys.exit(2)

    if args.ui:
        # UI mode streams height (altitude) by default.
        run_ui(args.mac, hci=args.hci, altitude=True, csv_path=args.csv)
        return
    if args.webui:
        run_webui(
            args.mac,
            hci=args.hci,
            csv_path=args.csv or "height.csv",
            port=args.webui_port,
            window_s=args.webui_window,
            open_browser=True,
        )
        return

    csv_file = None
    csv_writer = None
    plot_times_utc = None
    plot_values = None
    if args.csv:
        import os

        new_file = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        csv_file = open(args.csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if new_file:
            if args.altitude:
                csv_writer.writerow(["utc_iso", "altitude_m"])
            else:
                csv_writer.writerow(["utc_iso", "pressure_pa"])
        plot_times_utc = []
        plot_values = []

    device = MetaWear(args.mac, **kwargs)
    connect_device(device)

    board = device.board

    libmetawear.mbl_mw_baro_bosch_set_oversampling(board, BaroBoschOversampling.LOW_POWER)
    libmetawear.mbl_mw_baro_bosch_set_standby_time(board, 500.0)
    libmetawear.mbl_mw_baro_bosch_write_config(board)

    done = Event()
    callbacks = []

    deadline = monotonic() + args.seconds if args.seconds is not None else None
    measure = "Altitude" if args.altitude else "Pressure"
    bar_kw = {"unit": "sample", "desc": measure + (" (m)" if args.altitude else " (Pa)")}
    if args.seconds is None:
        bar_kw["total"] = args.samples
    bar = tqdm(**bar_kw)

    def data_handler(_ctx, data):
        val = parse_value(data)
        left = (
            "{:.1f}s left".format(max(0.0, deadline - monotonic())) if deadline is not None else ""
        )
        if args.altitude:
            bar.set_postfix(m="{:.2f}".format(val), t=left)
        else:
            bar.set_postfix(pa="{:.1f}".format(val), t=left)
        bar.update(1)
        if csv_writer is not None:
            ts = datetime.now(timezone.utc)
            csv_writer.writerow([ts.isoformat(), "{:.6f}".format(val)])
            csv_file.flush()
            plot_times_utc.append(ts)
            plot_values.append(float(val))
        if deadline is not None:
            if monotonic() >= deadline:
                done.set()
        elif bar.n >= args.samples:
            done.set()

    cb = FnVoid_VoidP_DataP(data_handler)
    callbacks.append(cb)

    if args.altitude:
        sig = libmetawear.mbl_mw_baro_bosch_get_altitude_data_signal(board)
    else:
        sig = libmetawear.mbl_mw_baro_bosch_get_pressure_data_signal(board)
    libmetawear.mbl_mw_datasignal_subscribe(sig, None, cb)
    libmetawear.mbl_mw_baro_bosch_start(board)

    try:
        while not done.is_set():
            if deadline is not None and monotonic() >= deadline:
                done.set()
                break
            sleep(0.05)
    finally:
        libmetawear.mbl_mw_datasignal_unsubscribe(sig)
        libmetawear.mbl_mw_baro_bosch_stop(board)
        bar.close()
        if csv_file is not None:
            csv_file.close()
            if plot_times_utc:
                save_csv_plot_session(args.csv, plot_times_utc, plot_values, args.altitude)
        device.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
