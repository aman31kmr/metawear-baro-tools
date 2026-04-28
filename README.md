# MetaWear barometer and IMU tools (USB/BLE)

Python helpers for MbientLab **MetaWear / MetaMotion** boards: barometer streaming and logging, **sensor-fusion IMU** (corrected acc / gyro / mag) to CSV with optional live web UI, **activity** hints (heuristic, few-shot model, or optional NLI zero-shot), and a small **labeled recording lab** for human-activity data collection.

The vendored SDK lives under `third_party/metawear-sdk-python/`. Run scripts from the **repo root** so imports resolve.

## Requirements

- **Python 3** on Linux (BLE used for streaming).
- System packages as needed for the MetaWear Python SDK (see MbientLab docs).
- Optional ML: `pip install -r requirements-har-ml.txt` (tqdm; sklearn/transformers only if you use those paths).

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
| `fewshot` | Needs `--activity-model PATH` to a JSON centroid model or sklearn joblib from training (below). |
| `nli_zero` | Optional HuggingFace NLI on a text summary of the window; install `transformers` + `torch` (slow on CPU). |

**Few-shot model from your labeled sessions**

```bash
python3 har_imu/train_fewshot_activity.py --labeled-root ./labeled_data --out ./models/activity_fewshot.json
python3 metawear_imu_stream.py <MAC> --webui --activity --activity-backend fewshot \
  --activity-model ./models/activity_fewshot.json --csv ./imu_run.csv
```

Optional RandomForest (only if scikit-learn works in your environment):

```bash
python3 har_imu/train_fewshot_activity.py --labeled-root ./labeled_data \
  --out ./models/activity_fewshot.json --sklearn-out ./models/activity_fewshot_rf.joblib
```

**Labeled recording (Tk UI)**

```bash
python3 har_imu/record_lab_ui.py <MAC> --out-dir ./labeled_data
```

Each session writes `session_<timestamp>/` with segment CSVs and `manifest.jsonl`.

## CSV helpers

- `imu_csv_format.py` — Parses legacy 5-column and 6-column (`epoch_ms`) IMU rows (and optional 7th `activity` column).
- `analyze_imu_csv.py` — Stats and sanity checks on captured IMU CSVs.

## Push to GitHub (feature branch)

HTTPS may return **403** if the PAT lacks **repo** write access or cached credentials are wrong. SSH is often easier:

```bash
chmod +x scripts/git_push_feature.sh
./scripts/git_push_feature.sh har-imu-labeled-stream
```

The script can create `~/.ssh/id_ed25519_metawear_github`, print the **public** key to add at [GitHub SSH keys](https://github.com/settings/keys), point `origin` at `git@github.com:aman31kmr/metawear-baro-tools.git`, and push `main` to the named branch. Pass a different branch name as the first argument. If you do not have push access, fork the repo, `git remote set-url origin git@github.com:<you>/metawear-baro-tools.git`, then run the same script.

## Example commands (one board MAC)

```bash
python3 metawear_baro_stream.py CF:96:FE:AD:63:E9 --altitude --seconds 30 --csv ./stairs.csv
python3 metawear_baro_stream.py CF:96:FE:AD:63:E9 --webui --csv ./height.csv
python3 metawear_imu_stream.py CF:96:FE:AD:63:E9 --webui --activity --csv ./imu_run.csv
python3 har_imu/record_lab_ui.py CF:96:FE:AD:63:E9 --out-dir ./labeled_data
```

Replace `CF:96:FE:AD:63:E9` with your board’s BLE address; on Linux use `--hci <adapter-mac>` when you have multiple adapters.
