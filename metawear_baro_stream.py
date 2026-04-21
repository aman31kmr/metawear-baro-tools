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

from tqdm import tqdm

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
    try:
        metawear_connect(device)
    except BaseException as err:
        print("Connect failed: {}".format(err), file=sys.stderr)
        raise
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
        default=None,
        metavar="PATH",
        help="Append UTC timestamp + value per row (pressure Pa or altitude m; header if new file)",
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
