#!/usr/bin/env python3
"""
Onboard logging for the Bosch barometer on MetaWear (no continuous BLE during capture).

How it works
------------
1. ``start`` — Connect once, configure the barometer, register a *data signal logger*, start
   logging + the sensor, then **disconnect**. The board keeps sampling and writing to **internal
   log memory** while you walk (no link required).

2. ``download`` — Connect again, stop + flush, find loggers via **mbl_mw_logger_lookup_id** (primary),
   fall back to anonymous discovery only if needed, pull entries over BLE, save CSV [+ optional PNG].

Limits
------
- Log capacity is finite; oldest entries may wrap when full (see Mbient docs for your model).
- Use ``--clear`` on ``start`` to erase previous onboard log data when you want a clean session.

Usage
-----
  python3 metawear_baro_log.py CF:96:FE:AD:63:E9 start --altitude
  python3 metawear_baro_log.py CF:96:FE:AD:63:E9 download --csv ./walk.csv --plot

  python3 metawear_baro_log.py CF:96:FE:AD:63:E9 start --altitude --clear

  # Halt logging/sensor without downloading (e.g. before a fresh start --clear)
  python3 metawear_baro_log.py CF:96:FE:AD:63:E9 stop

USB (MetaWear Python)
---------------------
The SDK auto-uses **USB serial** only for devices that enumerate as **VID:PID 1915:D978**
(MetaMotion S–class). Run ``usb-scan`` to see if Linux sees yours; if the list is empty,
the cable may be charge-only or the product is **BLE-only** — use Bluetooth + ``download``,
or export from the **MetaBase** app.

  python3 metawear_baro_log.py usb-scan
"""
from __future__ import print_function

import argparse
import csv
import os
import sys
from ctypes import byref, cast, POINTER, c_ubyte, c_void_p
from threading import Event
from time import monotonic, sleep

from tqdm import tqdm

from mbientlab.metawear import MetaWear, libmetawear, parse_value, create_voidp

from metawear_connect_safe import metawear_connect

from mbientlab.metawear.cbindings import (
    BaroBoschOversampling,
    FnVoid_VoidP_DataP,
    FnVoid_VoidP_UByte_Long_UByteP_UByte,
    FnVoid_VoidP_UInt_UInt,
    FnVoid_VoidP_VoidP_VoidP_UInt,
    LogDownloadHandler,
)


def connect_device(device):
    """
    Connect + SDK init. MetaWear chooses USB serial when ``MetaWearUSB.is_enumerated``,
    else Bluetooth — same MAC address in both cases.
    """
    print("Connecting ...", flush=True)
    metawear_connect(device)
    if not device.is_connected:
        raise RuntimeError("connect() returned but device.is_connected is False")
    if device.in_metaboot_mode:
        raise RuntimeError("Board is in MetaBoot (DFU) mode; exit DFU first.")
    if getattr(device.usb, "is_connected", False):
        print(
            "Transport: USB serial (best for log download).",
            flush=True,
        )
        sleep(0.3)
    else:
        print("Transport: Bluetooth LE", flush=True)
        libmetawear.mbl_mw_settings_set_connection_parameters(device.board, 7.5, 7.5, 0, 6000)
        sleep(1.5)
    print("Connected and ready:", device.address, flush=True)


def configure_baro_board(board):
    """Match `start` so module/signal state is consistent for logger lookup + download."""
    libmetawear.mbl_mw_baro_bosch_set_oversampling(board, BaroBoschOversampling.LOW_POWER)
    libmetawear.mbl_mw_baro_bosch_set_standby_time(board, 500.0)
    libmetawear.mbl_mw_baro_bosch_write_config(board)


def stop_collection(board, label=""):
    """
    Stop onboard logging and barometer so GATT traffic calms down before log discovery/download.
    Many boards never call the anonymous-datasignals callback while logging + sensor are active.
    """
    p = (" ({})".format(label)) if label else ""
    print("Stopping logging + barometer{} ...".format(p), flush=True)
    libmetawear.mbl_mw_logging_stop(board)
    libmetawear.mbl_mw_baro_bosch_stop(board)
    libmetawear.mbl_mw_logging_flush_page(board)


def cmd_stop(args):
    """Connect, stop logging + baro + flush, disconnect (no CSV)."""
    kwargs = {}
    if args.hci:
        kwargs["hci_mac"] = args.hci
    device = MetaWear(args.mac, **kwargs)
    connect_device(device)
    stop_collection(device.board, "halt capture")
    sleep(0.5)
    device.disconnect()
    print("Stopped. Log data on flash is preserved until you download or --clear.", flush=True)


def cmd_start(args):
    kwargs = {}
    if args.hci:
        kwargs["hci_mac"] = args.hci
    device = MetaWear(args.mac, **kwargs)
    connect_device(device)
    board = device.board

    if args.clear:
        print("Clearing previous onboard log entries ...", flush=True)
        libmetawear.mbl_mw_logging_clear_entries(board)
        sleep(0.5)

    configure_baro_board(board)

    if args.altitude:
        sig = libmetawear.mbl_mw_baro_bosch_get_altitude_data_signal(board)
    else:
        sig = libmetawear.mbl_mw_baro_bosch_get_pressure_data_signal(board)

    # Registers the logger on the board; must complete before logging_start.
    _baro_logger = create_voidp(
        lambda fn: libmetawear.mbl_mw_datasignal_log(sig, None, fn),
        resource="baro_logger",
    )

    print("Starting onboard logging + barometer ...", flush=True)
    libmetawear.mbl_mw_logging_start(board, 0)
    libmetawear.mbl_mw_baro_bosch_start(board)

    try:
        _lid = libmetawear.mbl_mw_logger_get_id(_baro_logger)
        print(
            "Logger id={} (after disconnect, download finds this via mbl_mw_logger_lookup_id).".format(
                _lid
            ),
            flush=True,
        )
    except Exception:
        pass

    print(
        "\nOnboard logging is active. You may disconnect BLE and move the board.\n"
        "When finished, bring it back and run:\n"
        "  python3 metawear_baro_log.py {} download --csv YOURFILE.csv{}\n".format(
            args.mac,
            " --altitude --plot" if args.altitude else " --plot",
        ),
        flush=True,
    )
    if not args.no_disconnect:
        device.disconnect()
        print("Disconnected (logging continues on the device).", flush=True)
    else:
        print("--no-disconnect: still connected; disconnect manually when ready.", flush=True)


def save_plot_for_csv(csv_path, altitude):
    """Write a PNG next to CSV: epoch_ms -> seconds from start, y = logged value."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    val_col = "altitude_m" if altitude else "pressure_pa"
    epochs = []
    vals = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Plot series", unit="pt"):
            if val_col not in row:
                val_col = "altitude_m" if "altitude_m" in row else "pressure_pa"
            epochs.append(float(row["epoch_ms"]))
            vals.append(float(row[val_col]))

    if not epochs:
        print("No rows to plot.", file=sys.stderr)
        return

    t0 = epochs[0]
    x_sec = [(t - t0) / 1000.0 for t in tqdm(epochs, desc="Time base", unit="pt", leave=False)]
    y_label = "Altitude (m)" if val_col == "altitude_m" else "Pressure (Pa)"
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.plot(x_sec, vals, color="#2563eb", linewidth=1.0, marker="o", markersize=2, alpha=0.85)
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel(y_label)
    ax.set_title(os.path.basename(csv_path))
    ax.grid(True, alpha=0.3)
    out = os.path.splitext(csv_path)[0] + ".png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved plot: {}".format(out), flush=True)


def _make_log_row_callback(rows, identifier):
    def on_data(_ctx, ptr):
        val = parse_value(ptr)
        rows.append((ptr.contents.epoch, float(val), identifier))

    return FnVoid_VoidP_DataP(on_data)


def cmd_download(args):
    # Keep ctypes callbacks alive for the whole download.
    _cb_keepalive = []

    kwargs = {}
    if args.hci:
        kwargs["hci_mac"] = args.hci
    device = MetaWear(args.mac, **kwargs)
    connect_device(device)
    board = device.board

    try:
        libmetawear.mbl_mw_metawearboard_set_time_for_response(board, 3000)
    except Exception:
        pass
    # Connection parameters already applied in connect_device() for BLE only.

    configure_baro_board(board)
    stop_collection(board, "prepare download")
    sleep(1.5)

    rows = []
    loggers = []

    if not args.anonymous_only:
        print(
            "Finding loggers with mbl_mw_logger_lookup_id (IDs 0–31). "
            "This is the supported way to reconnect to a saved log.",
            flush=True,
        )
        for lid in range(0, 32):
            ptr = libmetawear.mbl_mw_logger_lookup_id(board, c_ubyte(lid))
            if not ptr:
                continue
            pv = getattr(ptr, "value", None)
            if pv in (None, 0):
                continue
            print("  Logger id {} is present on the device.".format(lid), flush=True)
            loggers.append((lid, ptr))

    if not loggers:
        if args.anonymous_only:
            print("Using anonymous discovery only (--anonymous-only).", flush=True)
        else:
            print(
                "No logger handles via lookup (empty log or not yet written). "
                "Trying anonymous discovery (slow; may hang on some BLE stacks) ...",
                flush=True,
            )
        e_disc = Event()
        discovered = {"length": 0, "signals": None}

        def on_anonymous(_ctx, _bd, signals, n_sig):
            discovered["length"] = int(n_sig)
            n_loc = int(n_sig)
            discovered["signals"] = (
                cast(signals, POINTER(c_void_p * n_loc)) if signals is not None and n_loc > 0 else None
            )
            e_disc.set()

        handler_fn = FnVoid_VoidP_VoidP_VoidP_UInt(on_anonymous)
        _cb_keepalive.append(handler_fn)
        libmetawear.mbl_mw_metawearboard_create_anonymous_datasignals(board, None, handler_fn)
        deadline = monotonic() + args.discover_timeout
        while not e_disc.is_set():
            if monotonic() >= deadline:
                print(
                    "Anonymous discovery timed out. Plug the board via USB if available "
                    "(SDK uses USB automatically), update firmware, or export from MetaBase.",
                    file=sys.stderr,
                )
                device.disconnect()
                return 1
            e_disc.wait(timeout=5.0)
            if not e_disc.is_set():
                print(
                    "  ... anonymous discovery still waiting ({:.0f}s left) ...".format(
                        max(0.0, deadline - monotonic())
                    ),
                    flush=True,
                )

        n_sig = discovered["length"]
        sigs = discovered["signals"]
        if sigs is None or n_sig == 0:
            print("No onboard loggers found.", file=sys.stderr)
            device.disconnect()
            return 1

        print("Anonymous discovery: {} logger(s).".format(n_sig), flush=True)
        for i in range(n_sig):
            sig_ptr = sigs.contents[i]
            ident = libmetawear.mbl_mw_anonymous_datasignal_get_identifier(sig_ptr)
            id_str = ident.decode("utf-8", errors="replace")
            cb = _make_log_row_callback(rows, id_str)
            _cb_keepalive.append(cb)
            libmetawear.mbl_mw_anonymous_datasignal_subscribe(sig_ptr, None, cb)
            print("  Subscribed: {}".format(id_str), flush=True)

        n_dl = n_sig
    else:
        for lid, lptr in loggers:
            cb = _make_log_row_callback(rows, "logger_id_{}".format(lid))
            _cb_keepalive.append(cb)
            libmetawear.mbl_mw_logger_subscribe(lptr, None, cb)
            print("  Subscribed logger id {} for download.".format(lid), flush=True)

        n_dl = len(loggers)

    done = Event()
    dl_state = {"total": 0}

    def progress_update_handler(_ctx, entries_left, total_entries):
        try:
            total_entries = int(total_entries) if total_entries is not None else 0
            entries_left = int(entries_left) if entries_left is not None else 0
        except (TypeError, ValueError):
            return
        if total_entries > dl_state["total"]:
            dl_state["total"] = total_entries
        total = dl_state["total"] or total_entries
        if total > 0:
            done_cnt = total - entries_left
            pct = 100.0 * done_cnt / float(total) if total else 0.0
            print(
                "\rLog transfer: ~{}/{} entries (~{:.0f}%), {} left      ".format(
                    done_cnt, total, pct, entries_left
                ),
                end="",
                flush=True,
            )
        if entries_left == 0:
            print("", flush=True)
            print("Transfer finished (device reports 0 entries left).", flush=True)
            done.set()

    def unknown_entry_handler(_ctx, _entry_id, _epoch, _data, _length):
        print("\n[warn] unknown log entry — continuing", flush=True)

    progress_fn = FnVoid_VoidP_UInt_UInt(progress_update_handler)
    unknown_fn = FnVoid_VoidP_UByte_Long_UByteP_UByte(unknown_entry_handler)
    _cb_keepalive.extend([progress_fn, unknown_fn])
    download_handler = LogDownloadHandler(
        context=None,
        received_progress_update=progress_fn,
        received_unknown_entry=unknown_fn,
        received_unhandled_entry=cast(None, FnVoid_VoidP_DataP),
    )

    print("Pulling log over BLE (this can take a while) ...", flush=True)
    n_byte = n_dl if n_dl < 255 else 255
    libmetawear.mbl_mw_logging_download(board, n_byte, byref(download_handler))

    deadline_dl = monotonic() + args.download_timeout
    while not done.is_set():
        if monotonic() >= deadline_dl:
            print("\nDownload timed out.", file=sys.stderr)
            device.disconnect()
            return 1
        done.wait(timeout=2.0)

    libmetawear.mbl_mw_baro_bosch_stop(board)
    device.disconnect()
    print("Disconnected after download.", flush=True)

    if not rows:
        print("Download finished but received 0 data points.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r[0])
    col_val = "altitude_m" if args.altitude else "pressure_pa"
    new_file = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epoch_ms", col_val, "signal_id"])
        for epoch_ms, val, sid in tqdm(rows, desc="Writing CSV", unit="row"):
            w.writerow([epoch_ms, "{:.6f}".format(val), sid])
    print("Saved {}".format(args.csv), flush=True)

    if args.plot:
        save_plot_for_csv(args.csv, args.altitude)

    return 0


def cmd_usb_scan():
    """List boards visible on USB with the PID the MetaWear Python SDK uses."""
    from mbientlab.metawear import MetaWearUSB

    devs = MetaWearUSB.scan()
    if not devs:
        print(
            "No device found with VID:PID=1915:D978 (Mbient USB serial).\n"
            "- Try another USB cable (data lines).\n"
            "- Some MetaWear models have no USB device mode — use BLE only.\n"
            "- Check: lsusb | grep -i 1915",
            file=sys.stderr,
        )
        return 1
    print("MAC address\tname\tserial device", flush=True)
    for d in devs:
        print("{}\t{}\t{}".format(d["address"], d["name"], d["path"]), flush=True)
    print(
        "\nUse the MAC above, same as BLE, e.g.:\n"
        "  python3 metawear_baro_log.py {} download --csv ./out.csv --altitude".format(
            devs[0]["address"]
        ),
        flush=True,
    )
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "usb-scan":
        return cmd_usb_scan()

    parser = argparse.ArgumentParser(
        description="MetaWear barometer onboard log: start without tether, download later."
    )
    parser.add_argument("mac", help="BLE MAC of the board")
    parser.add_argument(
        "command",
        choices=["start", "download", "stop"],
        help="start | download (pull log to --csv) | stop (halt logging+sensors, no download)",
    )
    parser.add_argument(
        "--altitude",
        action="store_true",
        help="Log altitude (m) instead of pressure (Pa); use the same for start and download",
    )
    parser.add_argument("--hci", default=None, help="HCI adapter BLE MAC (Linux)")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="With start: erase onboard log memory before this session",
    )
    parser.add_argument(
        "--no-disconnect",
        action="store_true",
        help="With start: keep BLE up (testing only)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        metavar="PATH",
        help="With download: output CSV path",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="With download: also write a .png next to the CSV",
    )
    parser.add_argument(
        "--discover-timeout",
        type=float,
        default=120.0,
        metavar="SEC",
        help="download: max seconds to wait for anonymous logger discovery (default: 120)",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=600.0,
        metavar="SEC",
        help="download: max seconds to wait for log BLE transfer (default: 600)",
    )
    parser.add_argument(
        "--anonymous-only",
        action="store_true",
        help="download: skip mbl_mw_logger_lookup_id and use anonymous discovery only (debug / legacy)",
    )
    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
        return 0
    if args.command == "stop":
        cmd_stop(args)
        return 0
    if args.command == "download":
        if args.csv is None:
            print("download requires --csv PATH", file=sys.stderr)
            return 2
        return cmd_download(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
