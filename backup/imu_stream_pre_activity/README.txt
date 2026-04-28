IMU streaming backup (created when activity / har_imu was added to the main tree).

Contents
---------
metawear_imu_stream_snapshot_with_activity.py
  Exact copy of metawear_imu_stream.py from the project root at backup time
  (includes --webui --activity, /activity JSON, har_imu integration).

metawear_imu_stream_no_activity.py
  Runnable backup without activity UI: no --activity flag, no har_imu import,
  no /activity route. Same CSV epoch_ms, plot, and three-chart web UI otherwise.

imu_csv_format.py, analyze_imu_csv.py
  Copies of the CSV parser helper and analyzer at backup time.

har_imu_snapshot/
  Copy of the har_imu package at backup time.

Restore
-------
To replace the root script with the no-activity version (example):
  cp backup/imu_stream_pre_activity/metawear_imu_stream_no_activity.py metawear_imu_stream.py

Or keep the root file as-is and run the backup directly from repo root:
  python3 backup/imu_stream_pre_activity/metawear_imu_stream_no_activity.py MAC --webui --csv out.csv
