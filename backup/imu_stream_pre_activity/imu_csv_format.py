# Shared parsing for IMU long-format CSV (handles 5-col legacy and 6-col with epoch_ms).


def parse_imu_csv_row(parts):
    """
    Parse one CSV row (list of fields) into a dict or None.

    New format: utc_iso, epoch_ms, sensor, x, y, z
    Legacy:      utc_iso, sensor, x, y, z
    """
    if len(parts) < 5:
        return None
    s1 = parts[1].strip().lower()
    s2 = parts[2].strip().lower() if len(parts) > 2 else ""

    if len(parts) >= 6:
        try:
            epoch_ms = int(float(parts[1]))
        except (TypeError, ValueError):
            epoch_ms = None
        if epoch_ms is not None and s2 in ("acc", "gyro", "mag"):
            try:
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            except ValueError:
                return None
            return {
                "utc_iso": parts[0].strip(),
                "epoch_ms": epoch_ms,
                "sensor": s2,
                "x": x,
                "y": y,
                "z": z,
            }

    if s1 in ("acc", "gyro", "mag"):
        try:
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            return None
        return {
            "utc_iso": parts[0].strip(),
            "epoch_ms": None,
            "sensor": s1,
            "x": x,
            "y": y,
            "z": z,
        }
    return None
