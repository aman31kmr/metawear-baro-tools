# MetaWear barometer and IMU tools (USB/BLE)

Python helpers for MbientLab **MetaWear / MetaMotion** boards: barometer streaming and logging, **sensor-fusion IMU** (corrected acc / gyro / mag) to CSV with optional live web UI, **activity** recognition (heuristic + stats-threshold), and a small **labeled recording lab** for human-activity data collection.

The vendored SDK lives under `third_party/metawear-sdk-python/`. Run scripts from the **repo root** so imports resolve.

## Requirements

- **Python 3** on Linux (BLE used for streaming).
- System packages as needed for the MetaWear Python SDK (see MbientLab docs).
- Optional ML: `pip install -r requirements-har-ml.txt` (tqdm; sklearn/transformers only if you use those paths).

## Branches

This repo uses feature branches for experiments. Recommended starting points:

- **`har-imu-labeled-stream`**: activity recognition finalized around `--activity-backend stats` (robust SD + persistence) and labeled-data tooling.
- **`turn_detection`**: recording UI updated to collect `walking/running` with `straight/turn_left/turn_right` sublabels (in progress).
- **`4_sensors`**: `metawear_imu_stream.py` streams **acc/gyro/mag + barometer** in the same web UI + CSV (temporary experiment).

## Barometer

| Script | Purpose |
|--------|---------|
| `metawear_baro_stream.py` | Live stream to terminal; optional `--csv`, `--webui`, `--altitude`, PNG on stop. |
| `metawear_baro_log.py` | Onboard log, download to CSV, `--altitude`, `--plot`; `usb-scan` for USB class. |
| `metawear_connect_safe.py` | Safer connect when USB/BLE raises odd failures. |

**Quick examples**

```bash
python3 metawear_baro_stream.py <MAC> --altitude --seconds 30 --csv ./stairs.csv
python3 metawear_baro_log.py <MAC> start --altitude --clear
python3 metawear_baro_log.py <MAC> download --csv ./out.csv --altitude --plot
python3 metawear_baro_log.py usb-scan
```

## IMU (sensor fusion)

| Script | Purpose |
|--------|---------|
| `metawear_imu_stream.py` | Streams corrected acc, gyro, mag; `--webui` for three live charts; CSV uses `utc_iso`, `epoch_ms`, `sensor`, `x`, `y`, `z`. |
| `metawear_imu_stream_no_activity.py` | Snapshot variant without activity UI (see `backup/` for related copies). |

**Web UI + activity strip**

```bash
python3 metawear_imu_stream.py <MAC> --webui --activity --csv ./imu_run.csv
```

Activity JSON is served at `GET /activity` (about 1 s refresh from the UI). Backends:

| `--activity-backend` | Behavior |
|----------------------|----------|
| `heuristic` (default) | Rules on linear-acc magnitude + cadence band (`har_imu/realtime_estimator.py`). |
| `stats` | Stats-threshold model using **robust acc magnitude SD** + persistence gating (train via `har_imu/train_stats_activity.py`). |

### Activity recognition: what’s implemented

All activity backends share the same **streaming integration**:

- IMU samples arrive from sensor-fusion callbacks in `IMUStreamer._make_handler()` (`metawear_imu_stream.py`).
- For each `acc` + `gyro` row, we call `activity.push(sensor, epoch_ms, x, y, z)` and periodically `activity.maybe_recompute(...)`.
- The web UI polls `GET /activity` every ~1s and renders the returned JSON.

Backends:

- **Heuristic (`--activity-backend heuristic`)**
  - Code: `har_imu/realtime_estimator.py`
  - Uses a sliding window of **linear-acc magnitude** + FFT band energy (cadence proxy) + a vote smoother.

- **Stats-threshold (`--activity-backend stats`)** (recommended for “fast + robust” live use)
  - Train: `har_imu/train_stats_activity.py` → `models/activity_stats.json`
    - Computes **windowed acc magnitude SD** per activity from labeled CSVs.
    - Learns two thresholds: **standing→walking** and **walking→running**.
    - Default walk→run threshold is **quantile-biased** (midpoint of \(p90(walk)\) and \(p10(run)\)) to detect running earlier.
  - Stream: `har_imu/stats_threshold_estimator.py`
    - Computes **robust SD** by clipping within-window magnitudes to **p5..p95** (winsorization) so brief spikes/outliers don’t dominate.
    - Adds **persistence gating** so the label only changes after it’s sustained:
      - requires a short **streak** of consistent raw predictions
      - enforces a minimum **dwell time** in the current state
    - This makes “short spike” artifacts (like you created in `imu_run3.csv`) much less likely to flip the activity.

**Stats-threshold model (fastest live path; based on acc magnitude SD)**

```bash
python3 har_imu/train_stats_activity.py --labeled-root ./labeled_data --out ./models/activity_stats.json
python3 metawear_imu_stream.py <MAC> --webui --activity --activity-backend stats \
  --activity-model ./models/activity_stats.json --csv ./imu_run.csv
```

To make it switch to **running earlier**, lower the walk→run threshold by changing the quantiles:

```bash
python3 har_imu/train_stats_activity.py --labeled-root ./labeled_data --out ./models/activity_stats.json \
  --walk-run-lo-q 80 --walk-run-hi-q 10
```

**Labeled recording (Tk UI)**

```bash
python3 har_imu/record_lab_ui.py <MAC> --out-dir ./labeled_data
```

Each session writes `session_<timestamp>/` with segment CSVs and `manifest.jsonl`.

## CSV helpers

- `imu_csv_format.py` — Parses legacy 5-column and 6-column (`epoch_ms`) IMU rows (and optional 7th `activity` column).
- `analyze_imu_csv.py` — Stats and sanity checks on captured IMU CSVs.

## Example commands (one board MAC)

```bash
python3 metawear_baro_stream.py CF:96:FE:AD:63:E9 --altitude --seconds 30 --csv ./stairs.csv
python3 metawear_baro_stream.py CF:96:FE:AD:63:E9 --webui --csv ./height.csv
python3 metawear_imu_stream.py CF:96:FE:AD:63:E9 --webui --activity --csv ./imu_run.csv
python3 har_imu/record_lab_ui.py CF:96:FE:AD:63:E9 --out-dir ./labeled_data
```

Replace `CF:96:FE:AD:63:E9` with your board’s BLE address; on Linux use `--hci <adapter-mac>` when you have multiple adapters.
