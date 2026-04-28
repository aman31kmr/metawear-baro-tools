# MetaWear barometer logging (USB/BLE)

This repo contains small Python scripts to stream or onboard-log the Bosch barometer on MbientLab MetaWear / MetaMotion boards.

## Scripts

- `metawear_baro_stream.py`: live streaming to terminal + optional CSV/PNG.
- `metawear_baro_log.py`: onboard logging (walk out of range) + later download to CSV/PNG.
- `metawear_connect_safe.py`: safe connect wrapper (works around SDK raising non-exceptions on USB failure).

## Quick start

Streaming:

```bash
python3 metawear_baro_stream.py <MAC> --altitude --seconds 30 --csv ./stairs.csv
```

Onboard logging:

```bash
python3 metawear_baro_log.py <MAC> start --altitude --clear
# walk around, then later:
python3 metawear_baro_log.py <MAC> download --csv ./out.csv --altitude --plot
```

USB detection (MetaMotionS class):

```bash
python3 metawear_baro_log.py usb-scan
```

## Commands you ran (example session)

```bash
python3 metawear_baro_stream.py CF:96:FE:AD:63:E9 --altitude --seconds 30 --csv ./stair2s2.csv
python metawear_baro_stream.py CF:96:FE:AD:63:E9 --webui --csv height.csv

python3 /home/aman/metawear_baro_log.py CF:96:FE:AD:63:E9 start --altitude --clear
python3 /home/aman/metawear_baro_log.py CF:96:FE:AD:63:E9 stop
python3 /home/aman/metawear_baro_log.py CF:96:FE:AD:63:E9 download --csv ./out_room.csv --altitude --plot
```

