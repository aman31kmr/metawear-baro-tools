#!/usr/bin/env python3
"""
Stream accelerometer, gyroscope, and magnetometer via MetaWear sensor fusion
(corrected acc / gyro / mag vectors). Live web UI shows three panels; CSV rows are
appended as samples arrive; on stop, a three-panel PNG is written next to the CSV.

Each fusion output is a separate data stream — row counts ``n_acc`` / ``n_gyro`` / ``n_mag``
may differ slightly; use the SDK ``epoch_ms`` column for a common board-side time axis.
Host ``utc_iso`` is when the BLE callback ran (latency/jitter versus the board).

Requires firmware with sensor fusion (typical MetaMotion R / S / etc.).

Usage:
  python3 metawear_imu_stream.py --scan --scan-seconds 8
  python3 metawear_imu_stream.py CF:96:FE:AD:63:E9 --webui --csv ./run_imu.csv
  python3 metawear_imu_stream.py CF:96:FE:AD:63:E9 --webui --activity --csv ./run_imu.csv
  python3 har_imu/train_stats_activity.py --labeled-root ./labeled_data --out ./models/activity_stats.json
  python3 metawear_imu_stream.py CF:96:FE:AD:63:E9 --webui --activity --activity-backend stats \\
      --activity-model ./models/activity_stats.json --csv ./run_imu.csv
  python3 metawear_imu_stream.py CF:96:FE:AD:63:E9 --seconds 30 --csv ./run_imu.csv
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import signal
import sys
import traceback
import webbrowser
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread
from time import monotonic, sleep

_INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MetaWear IMU Live</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; color: #111827; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .box { border: 1px solid #d1d5db; border-radius: 12px; padding: 12px; background: #fafafa; }
    .box h3 { margin: 0 0 8px 0; font-size: 15px; }
    .vals { font-family: ui-monospace, monospace; font-size: 13px; line-height: 1.5; color: #1f2937; }
    .pill { display: inline-block; background: #e5e7eb; padding: 4px 8px; border-radius: 999px; font-size: 12px; margin-right: 6px; }
    canvas { width: 100%; height: 200px; border: 1px solid #d1d5db; border-radius: 8px; background: #fff; margin-top: 8px; }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
    .muted { color: #6b7280; font-size: 13px; }
    button { background: #ef4444; border: 0; color: white; padding: 8px 12px; border-radius: 10px; cursor: pointer; font-weight: 600; }
    button:disabled { background: #fca5a5; cursor: not-allowed; }
  </style>
</head>
<body>
  <h2 style="margin: 0 0 10px 0;">MetaWear IMU (live)</h2>
  <div class="row">
    <button id="stopBtn" title="Stops streaming and saves CSV + PNG">Stop</button>
    <span class="muted">Separate fusion streams → n differs slightly. Charts use board <code>epoch_ms</code> time (see CSV).</span>
  </div>
  <!--INJECT_ACTIVITY-->
  <div class="grid">
    <div class="box">
      <h3>Accelerometer</h3>
      <div><span class="pill">n=<span id="n_acc">0</span></span></div>
      <div class="vals" id="txt_acc">—</div>
      <canvas id="c_acc" width="600" height="200"></canvas>
    </div>
    <div class="box">
      <h3>Gyroscope</h3>
      <div><span class="pill">n=<span id="n_gyro">0</span></span></div>
      <div class="vals" id="txt_gyro">—</div>
      <canvas id="c_gyro" width="600" height="200"></canvas>
    </div>
    <div class="box">
      <h3>Magnetometer</h3>
      <div><span class="pill">n=<span id="n_mag">0</span></span></div>
      <div class="vals" id="txt_mag">—</div>
      <canvas id="c_mag" width="600" height="200"></canvas>
    </div>
  </div>

<script>
function drawChart(canvas, t, m, yLabel) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const ml=48, mr=12, mt=14, mb=28;
  const iw = Math.max(10, w-ml-mr), ih = Math.max(10, h-mt-mb);
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(ml, h-mb); ctx.lineTo(w-mr, h-mb);
  ctx.moveTo(ml, mt); ctx.lineTo(ml, h-mb);
  ctx.stroke();
  if (!t.length) {
    ctx.fillStyle = '#6b7280'; ctx.font = '12px system-ui';
    ctx.fillText('waiting…', ml+6, mt+16);
    return;
  }
  const t0 = t[0], t1 = t[t.length-1] > t0 ? t[t.length-1] : t0 + 1e-6;
  let y0 = Math.min(...m), y1 = Math.max(...m);
  if (y1 === y0) { y0 -= 0.5; y1 += 0.5; }
  else { const pad = Math.max(1e-6, 0.12*(y1-y0)); y0 -= pad; y1 += pad; }
  const sx = (x)=> ml + (x - t0) * iw / (t1 - t0);
  const sy = (v)=> mt + (y1 - v) * ih / (y1 - y0);
  ctx.strokeStyle = '#eef2f7';
  for (const frac of [0.25,0.5,0.75]) {
    const yy = mt + frac*ih;
    ctx.beginPath(); ctx.moveTo(ml, yy); ctx.lineTo(w-mr, yy); ctx.stroke();
  }
  ctx.fillStyle = '#6b7280'; ctx.font = '11px system-ui';
  ctx.fillText('0s', ml, h-8);
  ctx.textAlign = 'right';
  ctx.fillText((t1-t0).toFixed(1)+'s', w-mr, h-8);
  ctx.textAlign = 'left';
  ctx.fillText(yLabel, 4, mt+10);
  ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 1.8;
  ctx.beginPath();
  for (let i=0;i<t.length;i++){
    const xx=sx(t[i]), yy=sy(m[i]);
    if (i===0) ctx.moveTo(xx,yy); else ctx.lineTo(xx,yy);
  }
  ctx.stroke();
  const lx = sx(t[t.length-1]), ly = sy(m[m.length-1]);
  ctx.fillStyle = '#2563eb';
  ctx.beginPath(); ctx.arc(lx, ly, 3.2, 0, 2*Math.PI); ctx.fill();
}

const nextFrom = { acc: 0, gyro: 0, mag: 0 };
const buf = { acc: {t:[],m:[]}, gyro: {t:[],m:[]}, mag: {t:[],m:[]} };

async function tick(){
  try {
    const q = `acc=${nextFrom.acc}&gyro=${nextFrom.gyro}&mag=${nextFrom.mag}`;
    const r = await fetch(`/delta?${q}`, {cache:'no-store'});
    const j = await r.json();
    for (const k of ['acc','gyro','mag']) {
      const block = j[k];
      if (!block) continue;
      document.getElementById('n_'+k).textContent = block.n ?? 0;
      if (block.latest) {
        const v = block.latest;
        document.getElementById('txt_'+k).textContent =
          `x=${v[0].toFixed(4)}  y=${v[1].toFixed(4)}  z=${v[2].toFixed(4)}`;
      }
      if (block.t && block.m && block.t.length) {
        for (let i=0;i<block.t.length;i++) {
          buf[k].t.push(block.t[i]);
          buf[k].m.push(block.m[i]);
        }
        nextFrom[k] = block.next_from ?? nextFrom[k];
      }
    }
    const ylab = { acc: '|a| (g)', gyro: '|ω| (deg/s)', mag: '|B| (µT)' };
    drawChart(document.getElementById('c_acc'), buf.acc.t, buf.acc.m, ylab.acc);
    drawChart(document.getElementById('c_gyro'), buf.gyro.t, buf.gyro.m, ylab.gyro);
    drawChart(document.getElementById('c_mag'), buf.mag.t, buf.mag.m, ylab.mag);
  } catch(e) {}
  setTimeout(tick, 120);
}

tick();
__ACTIVITY_SCRIPT__
document.getElementById('stopBtn').addEventListener('click', async (ev)=>{
  ev.target.disabled = true;
  ev.target.textContent = 'Stopping...';
  try { await fetch('/stop', {method:'POST'}); ev.target.textContent = 'Stopped'; }
  catch(e) { ev.target.textContent = 'Stop failed'; }
});
</script>
</body>
</html>
"""

_ACTIVITY_PANEL_HTML = r"""
  <div class="row" style="align-items:flex-start;background:#f0fdf4;border:1px solid #6ee7b7;border-radius:12px;padding:14px;margin-bottom:8px;flex-wrap:wrap;">
    <div style="flex:1;min-width:220px;">
      <div style="font-size:12px;color:#065f46;font-weight:600;margin-bottom:6px;">Activity (heuristic, ~1s refresh)</div>
      <div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;">
        <span id="actLabel" style="font-size:26px;font-weight:700;color:#047857;">—</span>
        <span class="pill" id="actConf">conf —</span>
        <span class="pill" id="actFeat">window —</span>
      </div>
      <div class="muted" id="actHint" style="margin-top:8px;font-size:12px;line-height:1.4;">Waiting for buffer…</div>
    </div>
  </div>
""".strip()

_ACTIVITY_JS = r"""
async function pollActivity(){
  try{
    const r = await fetch('/activity', {cache:'no-store'});
    const j = await r.json();
    const lab = document.getElementById('actLabel');
    if(lab) lab.textContent = (j.label || 'unknown').toString().toUpperCase();
    const cf = document.getElementById('actConf');
    if(cf) cf.textContent = 'conf ' + (typeof j.confidence === 'number' ? j.confidence : '\u2014');
    const ft = document.getElementById('actFeat');
    if(ft) ft.textContent = (j.window_samples_acc || 0) + ' smpl / ' + (j.window_s || '') + 's';
    const hi = document.getElementById('actHint');
    if(hi){
      let tail = (j.updated_age_s != null) ? (' | label age ' + j.updated_age_s + 's') : '';
      hi.textContent = (j.detail || '') + tail;
    }
  }catch(e){}
  setTimeout(pollActivity, 1000);
}
pollActivity();
""".strip()


def build_index_html(enable_activity):
    html = _INDEX_HTML
    html = html.replace("<!--INJECT_ACTIVITY-->", _ACTIVITY_PANEL_HTML if enable_activity else "")
    html = html.replace("__ACTIVITY_SCRIPT__", _ACTIVITY_JS if enable_activity else "/* activity UI off */")
    return html


try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(iterable=None, **_kwargs):
        if iterable is None:
            class _Noop:
                n = 0

                def update(self, n=1):
                    self.n += 1

                def set_postfix(self, **_kw):
                    pass

                def close(self):
                    pass

            return _Noop()
        return iterable


from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import (
    FnVoid_VoidP_DataP,
    Model,
    Module,
    SensorFusionAccRange,
    SensorFusionData,
    SensorFusionGyroRange,
    SensorFusionMode,
)

from imu_csv_format import parse_imu_csv_row
from metawear_baro_stream import connect_device, run_scan


def _vec3(data):
    v = parse_value(data)
    return float(v.x), float(v.y), float(v.z)


def _mag(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


def save_imu_session_plot(csv_path):
    """Read long-format IMU CSV; write <stem>.png with three magnitude subplots."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    staged = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for parts in tqdm(reader, desc="Load CSV", unit="row"):
            if not parts or parts[0].strip().lower().startswith("utc_iso"):
                continue
            rec = parse_imu_csv_row(parts)
            if rec is None:
                continue
            try:
                name = rec["sensor"]
                ts = datetime.fromisoformat(rec["utc_iso"].replace("Z", "+00:00"))
                mm = _mag(rec["x"], rec["y"], rec["z"])
            except Exception:
                continue
            staged.append((name, ts, rec.get("epoch_ms"), mm))

    if not staged:
        print("No IMU rows to plot (empty or header-only CSV).", flush=True)
        return

    min_epoch_ms = None
    for _, _, ep, _ in staged:
        if ep is not None:
            min_epoch_ms = ep if min_epoch_ms is None else min(min_epoch_ms, ep)
    use_board_time = min_epoch_ms is not None

    series = {"acc": {"t": [], "m": []}, "gyro": {"t": [], "m": []}, "mag": {"t": [], "m": []}}
    t0_utc = {"acc": None, "gyro": None, "mag": None}

    for name, ts, epoch_ms, mm in staged:
        if use_board_time and epoch_ms is not None:
            tsec = (epoch_ms - min_epoch_ms) / 1000.0
        else:
            if t0_utc[name] is None:
                t0_utc[name] = ts
            tsec = (ts - t0_utc[name]).total_seconds()
        series[name]["t"].append(tsec)
        series[name]["m"].append(mm)

    if not series["acc"]["t"] and not series["gyro"]["t"] and not series["mag"]["t"]:
        print("No IMU rows to plot.", flush=True)
        return

    out_png = os.path.splitext(csv_path)[0] + ".png"
    titles = (
        ("Accelerometer |a| (g)", "acc"),
        ("Gyroscope |ω| (deg/s)", "gyro"),
        ("Magnetometer |B| (µT)", "mag"),
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        layout="constrained",
        sharex=bool(use_board_time),
    )
    xlab = (
        "Board time (s) from min epoch_ms (fusion clock; shared)"
        if use_board_time
        else "Wall time since first host-received sample in that stream (s)"
    )
    plot_tail_s = None
    if use_board_time:
        t_all = []
        for k in ("acc", "gyro", "mag"):
            t_all.extend(series[k]["t"])
        if t_all:
            span_all = max(t_all) - min(t_all)
            if span_all > 600.0:  # >10 minutes
                plot_tail_s = 180.0
                print(
                    "NOTE: {} spans {:.0f}s; plotting last {:.0f}s for readability.".format(
                        os.path.basename(csv_path),
                        span_all,
                        plot_tail_s,
                    ),
                    flush=True,
                )
    for ax, (title, key) in zip(axes, titles):
        td, md = series[key]["t"], series[key]["m"]
        if td:
            # Sort and break long gaps so separate sessions don't draw a straight line.
            pairs = sorted(zip(td, md), key=lambda p: p[0])
            td = [p[0] for p in pairs]
            md = [p[1] for p in pairs]
            if len(td) >= 8:
                dts = [td[i] - td[i - 1] for i in range(1, len(td)) if td[i] > td[i - 1]]
                if dts:
                    med_dt = sorted(dts)[len(dts) // 2]
                    gap_s = max(2.0, 50.0 * float(med_dt))
                else:
                    gap_s = 2.0
                td2, md2 = [td[0]], [md[0]]
                for i in range(1, len(td)):
                    if td[i] - td[i - 1] > gap_s:
                        td2.append(float("nan"))
                        md2.append(float("nan"))
                    td2.append(td[i])
                    md2.append(md[i])
                td, md = td2, md2

            if plot_tail_s is not None:
                t_clean = [t for t in td if not (isinstance(t, float) and t != t)]
                if t_clean:
                    t_hi = max(t_clean)
                    t_lo = t_hi - float(plot_tail_s)
                    td = [t if (isinstance(t, float) and t != t) or t >= t_lo else float("nan") for t in td]
                    md = [m if (isinstance(t, float) and t != t) or t >= t_lo else float("nan") for t, m in zip(td, md)]
            # Downsample for legible static PNGs; keep NaN gaps (don't reconnect lines).
            # Do it per contiguous segment separated by NaNs.
            def _downsample_keep_gaps(t, m, max_points=6000):
                seg_t, seg_m = [], []
                out_t, out_m = [], []
                def flush():
                    nonlocal seg_t, seg_m, out_t, out_m
                    if not seg_t:
                        return
                    n0 = len(seg_t)
                    stride = max(1, n0 // max_points)
                    out_t.extend(seg_t[::stride])
                    out_m.extend(seg_m[::stride])
                    seg_t, seg_m = [], []
                for ti, mi in zip(t, m):
                    if (isinstance(ti, float) and ti != ti) or (isinstance(mi, float) and mi != mi):
                        flush()
                        out_t.append(float("nan"))
                        out_m.append(float("nan"))
                        continue
                    seg_t.append(ti)
                    seg_m.append(mi)
                flush()
                return out_t, out_m

            td, md = _downsample_keep_gaps(td, md, max_points=6000)
            ax.plot(td, md, color="#2563eb", linewidth=1.1, alpha=0.9, rasterized=True)
        ax.set_ylabel(title.split()[1])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(xlab)
    if plot_tail_s is not None and use_board_time:
        try:
            t_max = max(max(series[k]["t"]) for k in ("acc", "gyro", "mag") if series[k]["t"])
            for ax in axes:
                ax.set_xlim(max(0.0, t_max - float(plot_tail_s)), t_max)
        except Exception:
            pass
    fig.suptitle(os.path.basename(csv_path), fontsize=11)
    fig.savefig(out_png, dpi=240)
    plt.close(fig)
    print("Saved plot: {}".format(out_png), flush=True)


class IMUStreamer:
    """Sensor fusion streams for corrected acc, gyro, mag; CSV append + ring buffers for UI."""

    def __init__(self, mac, *, hci=None, csv_path="imu.csv", series_maxlen=0, activity_estimator=None):
        # csv_path may be None to stream without writing until ``switch_recording_file`` is used.
        self.mac = mac
        self.hci = hci
        self.csv_path = csv_path
        self.series_maxlen = int(series_maxlen)
        self.device = None
        self.board = None
        self.signals = []
        self.callbacks = []
        self.done = Event()
        self.csv_file = None
        self.csv_writer = None
        self._csv_io_lock = Lock()
        # When set, each CSV row appends this activity label (for ML recording lab).
        self._activity_label_for_rows = None
        self._lock = Lock()
        self._make_series = lambda: (
            [] if self.series_maxlen <= 0 else deque(maxlen=self.series_maxlen)
        )
        self._series_acc = self._make_series()
        self._series_gyro = self._make_series()
        self._series_mag = self._make_series()
        # First board epoch_ms seen (any stream) — shared time axis for UI/plots vs host UTC.
        self._epoch_anchor_ms = None
        self.sample_count = {"acc": 0, "gyro": 0, "mag": 0}
        self.latest_xyz = {"acc": (0.0, 0.0, 0.0), "gyro": (0.0, 0.0, 0.0), "mag": (0.0, 0.0, 0.0)}
        self.activity = activity_estimator

    def _open_csv(self):
        """Open or append CSV; must not hold ``_csv_io_lock`` (acquires it internally)."""
        with self._csv_io_lock:
            self._open_csv_unlocked()

    def _open_csv_unlocked(self):
        if not self.csv_path:
            return
        new_file = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        if not new_file:
            with open(self.csv_path, "r", newline="") as rf:
                first = rf.readline()
            if "epoch_ms" not in first:
                # Auto-rotate to a new file instead of crashing: keeps legacy CSV intact.
                base, ext = os.path.splitext(self.csv_path)
                ext = ext or ".csv"
                cand = base + "_epoch" + ext
                k = 2
                while os.path.exists(cand) and os.path.getsize(cand) > 0:
                    cand = base + "_epoch{}".format(k) + ext
                    k += 1
                print(
                    "NOTE: {} has legacy 5-column header; writing new 6-column CSV to {}".format(
                        os.path.basename(self.csv_path),
                        os.path.basename(cand),
                    ),
                    flush=True,
                )
                self.csv_path = cand
                new_file = True
        self.csv_file = open(self.csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if new_file:
            header = ["utc_iso", "epoch_ms", "sensor", "x", "y", "z"]
            if self._activity_label_for_rows:
                header.append("activity")
            self.csv_writer.writerow(header)

    def switch_recording_file(self, new_path, *, activity_label=None):
        """
        Close the current CSV (if any) and start writing ``new_path``.
        Pass ``new_path=None`` to stop writing (no file) until the next switch.
        If ``activity_label`` is set (e.g. ``standing``), adds an ``activity`` column on each row.
        """
        with self._csv_io_lock:
            if self.csv_file is not None:
                try:
                    self.csv_file.close()
                except Exception:
                    pass
                self.csv_file = None
                self.csv_writer = None
            if not new_path:
                self.csv_path = None
                self._activity_label_for_rows = None
                return
            self.csv_path = new_path
            self._activity_label_for_rows = activity_label
            self._open_csv_unlocked()

    def _make_handler(self, name):
        def data_handler(_ctx, datap):
            epoch_ms = int(datap.contents.epoch)
            ts = datetime.now(timezone.utc)
            x, y, z = _vec3(datap)
            if self.csv_writer is not None:
                row = [
                    ts.isoformat(),
                    "{}".format(epoch_ms),
                    name,
                    "{:.6f}".format(x),
                    "{:.6f}".format(y),
                    "{:.6f}".format(z),
                ]
                if self._activity_label_for_rows:
                    row.append(self._activity_label_for_rows)
                with self._csv_io_lock:
                    if self.csv_writer is not None:
                        self.csv_writer.writerow(row)
                        self.csv_file.flush()
            mag = _mag(x, y, z)
            with self._lock:
                if self._epoch_anchor_ms is None:
                    self._epoch_anchor_ms = epoch_ms
                t_sec = (epoch_ms - self._epoch_anchor_ms) / 1000.0
                ser = getattr(self, "_series_" + name)
                ser.append((t_sec, mag))
                self.latest_xyz[name] = (x, y, z)
                self.sample_count[name] += 1

            if self.activity is not None and name in ("acc", "gyro"):
                self.activity.push(name, epoch_ms, x, y, z)
                self.activity.maybe_recompute(monotonic())

        return data_handler

    def start(self):
        kwargs = {}
        if self.hci:
            kwargs["hci_mac"] = self.hci
        self.device = MetaWear(self.mac, **kwargs)
        connect_device(self.device)
        self.board = self.device.board

        if libmetawear.mbl_mw_metawearboard_lookup_module(self.board, Module.SENSOR_FUSION) == Model.NA:
            raise RuntimeError("Sensor fusion is not available on this board / firmware.")

        libmetawear.mbl_mw_sensor_fusion_set_mode(self.board, SensorFusionMode.NDOF)
        libmetawear.mbl_mw_sensor_fusion_set_acc_range(self.board, SensorFusionAccRange._16G)
        libmetawear.mbl_mw_sensor_fusion_set_gyro_range(self.board, SensorFusionGyroRange._2000DPS)
        libmetawear.mbl_mw_sensor_fusion_write_config(self.board)
        sleep(0.2)

        libmetawear.mbl_mw_sensor_fusion_enable_data(self.board, SensorFusionData.CORRECTED_ACC)
        libmetawear.mbl_mw_sensor_fusion_enable_data(self.board, SensorFusionData.CORRECTED_GYRO)
        libmetawear.mbl_mw_sensor_fusion_enable_data(self.board, SensorFusionData.CORRECTED_MAG)

        pairs = [
            ("acc", SensorFusionData.CORRECTED_ACC),
            ("gyro", SensorFusionData.CORRECTED_GYRO),
            ("mag", SensorFusionData.CORRECTED_MAG),
        ]
        if self.csv_path:
            self._open_csv()
        for name, const in pairs:
            sig = libmetawear.mbl_mw_sensor_fusion_get_data_signal(self.board, const)
            self.signals.append(sig)
            cb = FnVoid_VoidP_DataP(self._make_handler(name))
            self.callbacks.append(cb)
            libmetawear.mbl_mw_datasignal_subscribe(sig, None, cb)

        libmetawear.mbl_mw_sensor_fusion_start(self.board)

    def stop(self):
        if self.done.is_set():
            return
        self.done.set()
        try:
            for sig, cb in zip(self.signals, self.callbacks):
                try:
                    libmetawear.mbl_mw_datasignal_unsubscribe(sig)
                except Exception:
                    pass
            try:
                if self.board is not None:
                    libmetawear.mbl_mw_sensor_fusion_stop(self.board)
            except Exception:
                pass
        finally:
            try:
                path = None
                with self._csv_io_lock:
                    if self.csv_file is not None:
                        self.csv_file.close()
                        self.csv_file = None
                        self.csv_writer = None
                    path = self.csv_path
                if path and os.path.exists(path) and os.path.getsize(path) > 64:
                    save_imu_session_plot(path)
            finally:
                if self.device is not None:
                    self.device.disconnect()
                    self.device = None
                self.board = None
                self.signals = []
                self.callbacks = []

    def snapshot_delta_triple(self, from_acc, from_gyro, from_mag):
        def pack(series, from_idx):
            n_total = len(series)
            if from_idx >= n_total:
                return {"t": [], "m": [], "next_from": n_total}
            chunk = (
                list(series[from_idx:n_total])
                if isinstance(series, list)
                else list(series)[from_idx:n_total]
            )
            t = [a for a, _ in chunk]
            m = [b for _, b in chunk]
            return {"t": t, "m": m, "next_from": n_total}

        with self._lock:
            latest = {
                "acc": list(self.latest_xyz["acc"]),
                "gyro": list(self.latest_xyz["gyro"]),
                "mag": list(self.latest_xyz["mag"]),
            }
            acc_b = pack(self._series_acc, from_acc)
            gyro_b = pack(self._series_gyro, from_gyro)
            mag_b = pack(self._series_mag, from_mag)
            counts = {
                "acc": self.sample_count["acc"],
                "gyro": self.sample_count["gyro"],
                "mag": self.sample_count["mag"],
            }
        return {
            "acc": {**acc_b, "latest": latest["acc"], "n": counts["acc"]},
            "gyro": {**gyro_b, "latest": latest["gyro"], "n": counts["gyro"]},
            "mag": {**mag_b, "latest": latest["mag"], "n": counts["mag"]},
        }


def run_webui(
    mac,
    *,
    hci=None,
    csv_path="imu.csv",
    port=8000,
    open_browser=True,
    enable_activity=False,
    activity_backend="heuristic",
    activity_model=None,
):
    activity_est = None
    if enable_activity:
        b = (activity_backend or "heuristic").strip().lower()
        if b in ("stats", "stats_threshold", "threshold"):
            if not activity_model:
                raise SystemExit("--activity-backend stats requires --activity-model PATH.json (from train_stats_activity.py)")
            from har_imu.stats_threshold_estimator import (
                StatsThresholdStreamEstimator,
                load_stats_model,
            )

            model = load_stats_model(activity_model)
            activity_est = StatsThresholdStreamEstimator(
                model,
                # Spike immunity defaults: winsorize + persistence
                clip_quantiles=(5.0, 95.0),
                min_state_s=2.5,
                require_streak=2,
            )
        else:
            from har_imu import RealtimeActivityEstimator

            activity_est = RealtimeActivityEstimator(window_s=14.0, min_recompute_s=1.0, vote_len=5)

    streamer = IMUStreamer(
        mac, hci=hci, csv_path=csv_path, series_maxlen=0, activity_estimator=activity_est
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

        def log_message(self, *_args):
            return

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index.html"):
                self._send(200, build_index_html(enable_activity))
                return
            if self.path.startswith("/activity"):
                if streamer.activity is None:
                    self._send(
                        200,
                        json.dumps(
                            {
                                "label": "off",
                                "confidence": 0.0,
                                "detail": "Start with --activity to enable this panel.",
                                "window_samples_acc": 0,
                                "window_s": 0,
                                "backend": "off",
                                "updated_age_s": None,
                            }
                        ),
                        content_type="application/json",
                    )
                    return
                streamer.activity.maybe_recompute(monotonic())
                self._send(200, json.dumps(streamer.activity.snapshot()), content_type="application/json")
                return
            if self.path.startswith("/delta"):
                try:
                    q = self.path.split("?", 1)[1] if "?" in self.path else ""
                    params = {}
                    for kv in q.split("&"):
                        if not kv or "=" not in kv:
                            continue
                        k, v = kv.split("=", 1)
                        params[k] = v
                    fa = int(params.get("acc", "0") or 0)
                    fg = int(params.get("gyro", "0") or 0)
                    fm = int(params.get("mag", "0") or 0)
                except Exception:
                    fa = fg = fm = 0
                snap = streamer.snapshot_delta_triple(fa, fg, fm)
                self._send(200, json.dumps(snap), content_type="application/json")
                return
            self._send(404, "not found", content_type="text/plain")

        def do_POST(self):
            if self.path.startswith("/stop"):
                self._send(200, "ok", content_type="text/plain")

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
    url = "http://{}:{}/".format(host, int(port))
    print("Web UI: {}".format(url), flush=True)
    if enable_activity:
        print(
            "Activity panel: GET /activity (backend={})".format(activity_backend or "heuristic"),
            flush=True,
        )
    if open_browser:
        try:
            webbrowser.open(url, new=1, autoraise=True)
        except Exception:
            pass

    def worker():
        try:
            streamer.start()
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
            Thread(target=httpd.shutdown, daemon=True).start()
        except Exception:
            pass

    old_int = old_term = None
    try:
        old_int = signal.signal(signal.SIGINT, _request_stop)
        old_term = signal.signal(signal.SIGTERM, _request_stop)
    except Exception:
        pass

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
            try:
                if old_int is not None:
                    signal.signal(signal.SIGINT, old_int)
                if old_term is not None:
                    signal.signal(signal.SIGTERM, old_term)
            except Exception:
                pass
        try:
            base = os.path.splitext(csv_path)[0]
            print("Saved CSV: {}".format(os.path.abspath(csv_path)), flush=True)
            print("Saved plot: {}".format(os.path.abspath(base + ".png")), flush=True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Stream IMU (acc / gyro / mag) from MetaWear via sensor fusion"
    )
    parser.add_argument("mac", nargs="?", help="BLE MAC of the board")
    parser.add_argument("--hci", default=None, help="HCI adapter BLE MAC (Linux)")
    parser.add_argument("--scan", action="store_true", help="Scan for BLE devices and exit")
    parser.add_argument("--scan-seconds", type=float, default=8.0, help="With --scan, listen time")
    parser.add_argument(
        "--csv",
        default="imu.csv",
        metavar="PATH",
        help=(
            "Append rows: utc_iso,epoch_ms,sensor,x,y,z (epoch = board fusion clock ms). PNG on stop."
        ),
    )
    parser.add_argument("--webui", action="store_true", help="Browser UI with three live charts")
    parser.add_argument(
        "--activity",
        action="store_true",
        help="With --webui: show standing/walking/running activity panel (~1s updates; see har_imu/)",
    )
    parser.add_argument(
        "--activity-backend",
        default="heuristic",
        metavar="NAME",
        help=(
            "With --activity: heuristic | stats. "
            "stats uses --activity-model PATH.json (train with har_imu/train_stats_activity.py). "
        ),
    )
    parser.add_argument(
        "--activity-model",
        default=None,
        metavar="PATH",
        help="With --activity-backend stats: JSON from train_stats_activity.py",
    )
    parser.add_argument("--webui-port", type=int, default=8000, help="Port for --webui")
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="Headless: run for this many seconds then save and exit (ignored with --webui)",
    )
    args = parser.parse_args()

    if args.scan:
        run_scan(args.hci, args.scan_seconds)
        return

    if not args.mac:
        print("Provide board MAC or use --scan.", file=sys.stderr)
        sys.exit(2)

    if args.webui:
        run_webui(
            args.mac,
            hci=args.hci,
            csv_path=args.csv or "imu.csv",
            port=args.webui_port,
            open_browser=True,
            enable_activity=args.activity,
            activity_backend=args.activity_backend,
            activity_model=args.activity_model,
        )
        return

    streamer = IMUStreamer(args.mac, hci=args.hci, csv_path=args.csv, series_maxlen=0)
    deadline = monotonic() + args.seconds if args.seconds is not None else None
    if deadline is None:
        print("Headless mode requires --seconds (or use --webui).", file=sys.stderr)
        sys.exit(2)

    bar = tqdm(unit="row", desc="IMU capture", dynamic_ncols=True)

    try:
        streamer.start()
        while monotonic() < deadline:
            sleep(0.08)
            with streamer._lock:
                n = sum(streamer.sample_count[k] for k in ("acc", "gyro", "mag"))
            bar.n = n
            bar.set_postfix(acc=streamer.sample_count["acc"], gyro=streamer.sample_count["gyro"], mag=streamer.sample_count["mag"])
            bar.refresh()
    except KeyboardInterrupt:
        pass
    finally:
        bar.close()
        streamer.stop()
        print("Disconnected.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
