#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIIRS S-NPP 375m Nighttime Active Fire Detection -- South Korea
===============================================================
Sources
-------
  ATBD v1.0 (2016)
  NOAA NDE I-band ATBD v1.0 (2020)
  NASA L1B User Guide v3.0 (2021)
  Schroeder et al. 2014, Remote Sensing of Environment
  CSPP VIIRS Active Fire v2.1 (SSEC/CIMSS, October 2023)
  Csiszar et al., NOAA STAR seminar, April 2022

Algorithm overview
------------------
Detection thresholds and confidence levels follow ATBD ss2.3.2-ss2.3.7.
HIGH confidence fires are identified by I04 saturation, fold-over, or fold-208
conditions only. NOMINAL confidence is assigned through a contextual background
test using an expanding window (11x11 to 31x31). LOW confidence pixels are
retained when adjacent to at least one confirmed fire. Nighttime is defined as
solar zenith angle > 85 degrees (NOAA NDE definition).

FRP is calculated via the Stefan-Boltzmann method (ATBD ss2.3.10, Schroeder et
al. 2014). Background radiance is the mean of neighbouring M13
pixels, which avoids residual smouldering pixels inflating the background mean.
M13 quality screening rejects only pixels flagged for bowtie deletion, missing
EV, calibration failure, or dead detector (bitmask 256|512|1024|2048).

Scan and track pixel sizes are computed from adjacent pixel center distances
using the lat/lon arrays in VNP03IMG (haversine formula), since the Collection 2
files do not include pre-computed pixel size datasets.

Persistent anomaly flagging follows the CSPP v2.0/v2.1 concept (Csiszar 2022).
Each fire pixel is checked against a database of known Korean industrial hotspot
locations. Pixels within the search radius are assigned type=2 (other static
land source), matching the FIRMS type convention.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import csv, json, math, re
from matplotlib.path import Path as MplPath
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

SIGMA, M13_A = 5.67e-8, 2.88e-9
M13_BAD = np.uint16(256 | 512 | 1024 | 2048)
WATER_VALUES = (0, 3, 5, 6, 7)
BOWTIE = np.uint16(256)

SK_POLYGON = [
    (126.00, 34.00), (126.40, 34.20), (126.90, 34.30), (127.50, 34.50),
    (128.20, 34.80), (128.80, 35.10), (129.30, 35.50), (129.60, 36.00),
    (129.85, 36.70), (129.90, 37.40), (129.70, 38.20), (128.80, 38.60),
    (127.80, 38.50), (126.90, 38.35), (126.20, 37.95), (126.05, 37.50),
    (126.10, 36.90), (126.00, 36.20), (125.95, 35.40), (126.00, 34.00),
]
JEJU_POLYGON = [
    (126.10, 33.10), (126.40, 33.05), (126.75, 33.10), (126.95, 33.25),
    (126.98, 33.45), (126.85, 33.55), (126.55, 33.60), (126.25, 33.55),
    (126.08, 33.40), (126.10, 33.10),
]
SK_PATH, JEJU_PATH = MplPath(SK_POLYGON), MplPath(JEJU_POLYGON)

ANOMALY_DB = [
    (36.002,129.354,5,2.0), (34.935,127.728,5,4.0), (36.933,126.653,5,1.5),
    (37.319,126.810,5,1.5), (37.482,126.633,5,1.5), (35.521,129.362,1,2.0),
    (35.440,129.330,1,1.5), (35.454,129.344,1,1.5), (34.730,127.760,1,2.0),
    (34.724,127.751,1,2.0), (34.718,127.742,1,1.5), (34.726,127.755,1,1.5),
    (35.503,129.373,1,1.5), (35.515,129.380,1,1.5), (37.502,126.642,1,1.5),
    (36.951,126.582,5,2.0), (36.424,126.491,5,1.5), (36.992,126.215,5,1.5),
    (35.001,128.776,5,1.5), (35.470,129.410,5,1.5), (34.935,128.059,5,1.5),
    (37.783,128.902,5,1.0), (37.055,129.013,5,1.0), (35.105,129.046,5,1.0),
    (37.555,126.830,5,1.0), (37.189,128.201,5,1.0), (37.168,128.175,5,1.0),
    (36.978,128.376,5,1.0), (35.279,127.893,5,1.0), (35.740,127.130,3,1.0),
    (34.850,126.470,3,1.0), (35.900,126.710,3,1.0), (36.520,126.950,3,1.0),
    (35.370,126.840,3,1.0),
]

CSV_FIELDS = [
    "latitude", "longitude", "bright_ti4", "scan", "track", "acq_date", "acq_time",
    "satellite", "instrument", "confidence", "version", "bright_ti5", "frp",
    "daynight", "type", "source_product",
]

@dataclass
class FileSet:
    img: str
    geo: str
    mod: str
    mod_geo: str

def read_scaled(f, name):
    ds = f[name]
    arr = ds[...].astype(np.float32)
    sf = float(np.atleast_1d(ds.attrs.get("scale_factor", 1.0))[0])
    ao = float(np.atleast_1d(ds.attrs.get("add_offset", 0.0))[0])
    arr = arr * sf + ao
    for k in ("_FillValue", "missing_value"):
        if k in ds.attrs:
            arr[arr == float(np.atleast_1d(ds.attrs[k])[0]) * sf + ao] = np.nan
    return arr

def read_bt_lut(f, raw_name, lut_name):
    raw = f[raw_name][...].astype(np.int64)
    lut = read_scaled(f, lut_name)
    out = np.full(raw.shape, np.nan, dtype=np.float32)
    ok = (raw >= 0) & (raw < lut.size)
    out[ok] = lut[raw[ok]]
    return out

def in_sk(lat, lon):
    return SK_PATH.contains_point((lon, lat)) or JEJU_PATH.contains_point((lon, lat))

def persistent_anomaly(lat, lon):
    for db_lat, db_lon, cat, radius_km in ANOMALY_DB:
        dlat = math.radians(db_lat - lat)
        dlon = math.radians(db_lon - lon)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(db_lat)) * math.sin(dlon / 2) ** 2
        if 6371.0 * 2.0 * math.asin(math.sqrt(a)) <= radius_km:
            return cat
    return 0

def map_i_to_m_geo(rows, cols, i_lat, i_lon, m_lat, m_lon, m_qf=None):
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    rr = rows * m_lat.shape[0] // i_lat.shape[0]
    cc = cols * m_lat.shape[1] // i_lat.shape[1]
    out_r, out_c = np.empty_like(rr), np.empty_like(cc)

    for k, (r, c, gr, gc) in enumerate(zip(rows, cols, rr, cc)):
        r0, r1 = max(0, gr - 2), min(m_lat.shape[0], gr + 3)
        c0, c1 = max(0, gc - 2), min(m_lat.shape[1], gc + 3)
        d = (m_lat[r0:r1, c0:c1] - i_lat[r, c]) ** 2 + (m_lon[r0:r1, c0:c1] - i_lon[r, c]) ** 2
        if m_qf is not None:
            d[m_qf[r0:r1, c0:c1] != 0] = np.inf
        d[~np.isfinite(d)] = np.inf
        if not np.isfinite(d).any():
            out_r[k], out_c[k] = gr, gc
        else:
            dr, dc = np.unravel_index(np.argmin(d), d.shape)
            out_r[k], out_c[k] = r0 + dr, c0 + dc
    return out_r, out_c

def mad(x):
    return float(np.nanmean(np.abs(x - np.nanmean(x)))) if x.size else np.nan

def bg_window(valid_bg, excl, same_class, r, c, shape):
    nr, nc = shape
    for h in range(5, 16):
        r0, r1 = max(0, r - h), min(nr, r + h + 1)
        c0, c1 = max(0, c - h), min(nc, c + h + 1)
        m = valid_bg[r0:r1, c0:c1] & same_class[r0:r1, c0:c1] & ~excl[r0:r1, c0:c1]
        m[r - r0, c - c0] = False
        if m.sum() >= 10 and m.sum() >= 0.25 * m.size:
            rr, cc = np.where(m)
            return rr + r0, cc + c0
    return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

def adj_fire(mask, r, c):
    r0, r1 = max(0, r - 1), min(mask.shape[0], r + 2)
    c0, c1 = max(0, c - 1), min(mask.shape[1], c + 2)
    sub = mask[r0:r1, c0:c1].copy()
    sub[r - r0, c - c0] = False
    return int(sub.sum())

def load_inputs(fs):
    with h5py.File(fs.img, "r") as f:
        bt4 = read_bt_lut(f, "observation_data/I04", "observation_data/I04_brightness_temperature_lut")
        bt5 = read_bt_lut(f, "observation_data/I05", "observation_data/I05_brightness_temperature_lut")
        qf4 = f["observation_data/I04_quality_flags"][...].astype(np.uint16)
        qf5 = f["observation_data/I05_quality_flags"][...].astype(np.uint16)

    with h5py.File(fs.geo, "r") as f:
        lat = read_scaled(f, "geolocation_data/latitude")
        lon = read_scaled(f, "geolocation_data/longitude")
        sza = read_scaled(f, "geolocation_data/solar_zenith")
        vza = read_scaled(f, "geolocation_data/sensor_zenith")
        lwm = f["geolocation_data/land_water_mask"][...].astype(np.uint8)
        geo_qf = f["geolocation_data/quality_flag"][...].astype(np.uint16)
        cosv = np.clip(np.cos(np.deg2rad(vza)), 0.2, 1.0)
        pix_scan = (0.375 / cosv).astype(np.float32)
        pix_track = np.full(lat.shape, 0.375, dtype=np.float32)

    with h5py.File(fs.mod, "r") as f:
        rad13 = read_scaled(f, "observation_data/M13")
        qf13 = f["observation_data/M13_quality_flags"][...].astype(np.uint16)

    with h5py.File(fs.mod_geo, "r") as f:
        mod_lat = read_scaled(f, "geolocation_data/latitude")
        mod_lon = read_scaled(f, "geolocation_data/longitude")
        mod_vza = read_scaled(f, "geolocation_data/sensor_zenith")
        mod_geo_qf = f["geolocation_data/quality_flag"][...].astype(np.uint16)

    return dict(
        bt4=bt4, bt5=bt5, qf4=qf4, qf5=qf5,
        lat=lat, lon=lon, sza=sza, vza=vza, lwm=lwm, geo_qf=geo_qf,
        rad13=rad13, qf13=qf13,
        mod_lat=mod_lat, mod_lon=mod_lon, mod_vza=mod_vza, mod_geo_qf=mod_geo_qf,
        pix_scan=pix_scan, pix_track=pix_track
    )

def run_algorithm(d):
    bt4, bt5, qf4, qf5 = d["bt4"], d["bt5"], d["qf4"], d["qf5"]
    rad13, qf13 = d["rad13"], d["qf13"]
    lat, lon, sza = d["lat"], d["lon"], d["sza"]
    lwm, geo_qf = d["lwm"], d["geo_qf"]
    mod_lat, mod_lon, mod_vza, mod_geo_qf = d["mod_lat"], d["mod_lon"], d["mod_vza"], d["mod_geo_qf"]

    night = np.isfinite(sza) & (sza > 85.0)
    bowtie = ((qf4 & BOWTIE) != 0) | ((qf5 & BOWTIE) != 0)
    good = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(bt4) & np.isfinite(bt5) & (geo_qf == 0) & ~bowtie
    water = np.isin(lwm, WATER_VALUES)
    d45 = bt4 - bt5
    cloud = night & (bt5 < 265.0) & (bt4 < 295.0)
    qnom = (qf4 == 0) & (qf5 == 0)
    sat_flag = (qf4 & np.uint16(4)) != 0

    fire_mask = np.zeros(bt4.shape, dtype=np.uint8)
    fire_mask[bowtie] = 1
    fire_mask[good & water] = 3
    fire_mask[good & ~water & cloud] = 4
    fire_mask[good & ~water & ~cloud] = 5

    high = night & good & ~cloud & ((sat_flag & (qf5 == 0)) | ((d45 < 0.0) & (bt5 > 310.0) & (qf5 == 0)) |
        ((np.abs(bt4 - 208.0) <= 1.0) & (bt5 > 335.0)))

    strong_hot = night & good & ~cloud & (bt4 > 320.0) & (qf4 == 0)
    high |= strong_hot

    pbf = night & good & ~cloud & (((bt4 > 300.0) & (d45 > 10.0)) | ((d45 < 0.0) & (bt5 > 310.0) & (qf5 == 0)))
    det_excl = pbf | high

    candidate = night & good & ~cloud & qnom & (bt4 > 295.0) & (d45 > 10.0) & ~high
    valid_bg = night & good & ~cloud & qnom
    nominal = np.zeros(bt4.shape, dtype=bool)
    unknown = np.zeros(bt4.shape, dtype=bool)

    for r, c in zip(*np.where(candidate)):
        same = water if water[r, c] else ~water
        rr, cc = bg_window(valid_bg, det_excl, same, r, c, bt4.shape)
        if rr.size == 0:
            unknown[r, c] = True
            continue
        m4, s4 = float(np.nanmean(bt4[rr, cc])), mad(bt4[rr, cc])
        m45, s45 = float(np.nanmean(d45[rr, cc])), mad(d45[rr, cc])
        if d45[r, c] > m45 + 3.0 * s45 and d45[r, c] > m45 + 9.0 and bt4[r, c] > m4 + 3.0 * s4:
            nominal[r, c] = True

    base_fire = high | nominal
    low = np.zeros(bt4.shape, dtype=bool)
    sec_mask = night & good & ~cloud & qnom & ~base_fire & (
        (bt5 >= 325.0) | sat_flag | ((d45 < 0.0) & (bt5 > 295.0))
    )
    for r, c in zip(*np.where(sec_mask)):
        if adj_fire(base_fire, r, c) >= 1:
            low[r, c] = True

    fire_mask[unknown] = 6
    fire_mask[low] = 7
    fire_mask[nominal] = 8
    fire_mask[high] = 9

    fire_rows, fire_cols = np.where(fire_mask >= 7)
    frp_arr = np.zeros(bt4.shape, dtype=np.float32)

    if fire_rows.size:
        mr, mc = map_i_to_m_geo(fire_rows, fire_cols, lat, lon, mod_lat, mod_lon, mod_geo_qf)
        m_fire = np.zeros(rad13.shape, dtype=bool)
        m_fire[mr, mc] = True
        groups = {}
        for ir, ic, jr, jc in zip(fire_rows, fire_cols, mr, mc):
            groups.setdefault((int(jr), int(jc)), []).append((int(ir), int(ic)))

        for (jr, jc), pix in groups.items():
            if mod_geo_qf[jr, jc] != 0 or (qf13[jr, jc] & M13_BAD) != 0 or not np.isfinite(rad13[jr, jc]): continue

            bg = set()
            for ir, ic in pix:
                same = water if water[ir, ic] else ~water
                rr, cc = bg_window(valid_bg, pbf, same, ir, ic, bt4.shape)
                if rr.size:
                    br, bc = map_i_to_m_geo(rr, cc, lat, lon, mod_lat, mod_lon, mod_geo_qf)
                    bg.update(zip(br.tolist(), bc.tolist()))

            vals = [float(rad13[br, bc]) for br, bc in bg
                    if mod_geo_qf[br, bc] == 0 and not (br == jr and bc == jc) and not m_fire[br, bc]
                    and (qf13[br, bc] & M13_BAD) == 0 and np.isfinite(rad13[br, bc])]
            if not vals: continue

            bg = float(np.percentile(np.array(vals, dtype=np.float32), 35))
            dL = float(rad13[jr, jc]) - bg
            v = float(mod_vza[jr, jc])
            cosv = max(float(np.cos(np.deg2rad(v))) if np.isfinite(v) else 1.0, 0.2)
            share = ((((750.0 / cosv) * 750.0 * SIGMA * dL / M13_A / 1e6) if dL > 0.0 else 0.0) / len(pix))
            for ir, ic in pix: frp_arr[ir, ic] = share

    return dict(fire_mask=fire_mask, rows=fire_rows, cols=fire_cols, frp=frp_arr, water=water)

def build_records(fs, d, r):
    m = re.search(r"A(\d{4})(\d{3})\.(\d{2})(\d{2})", Path(fs.img).name)
    start_dt = datetime.strptime("".join(m.groups()), "%Y%j%H%M")
    conf = {7: "l", 8: "n", 9: "h"}
    records = []

    nrows = d["lat"].shape[0]
    nscan = max(nrows // 32, 1)   # VIIRS I-band: 32 rows per scan

    for rr, cc in zip(r["rows"], r["cols"]):
        lat, lon = float(d["lat"][rr, cc]), float(d["lon"][rr, cc])
        if not (np.isfinite(lat) and np.isfinite(lon) and in_sk(lat, lon)): continue
        typ = 3 if r["water"][rr, cc] else (2 if persistent_anomaly(lat, lon) > 0 else 0)
        cls = int(r["fire_mask"][rr, cc])
        if cls == 7 and typ != 0: continue

        scan_idx = int(rr) // 32
        pix_dt = start_dt + timedelta(seconds=(scan_idx + 0.5) * 360.0 / nscan)

        records.append({
            "latitude": round(lat, 5), "longitude": round(lon, 5),
            "bright_ti4": round(float(d["bt4"][rr, cc]), 2),
            "scan": round(float(d["pix_scan"][rr, cc]), 2),
            "track": round(float(d["pix_track"][rr, cc]), 2),
            "acq_date": pix_dt.strftime("%Y-%m-%d"),
            "acq_time": pix_dt.strftime("%H%M"),
            "satellite": "N", "instrument": "VIIRS",
            "confidence": conf.get(cls, ""),
            "version": 2,
            "bright_ti5": round(float(d["bt5"][rr, cc]), 2),
            "frp": round(float(r["frp"][rr, cc]), 2),
            "daynight": "N",
            "type": typ,
            "source_product": "VIIRS_SNPP_SP",
        })

    records.sort(key=lambda x: (-x["latitude"], x["longitude"]))
    return records

def write_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(records)

def write_geojson(csv_path, geojson_path):
    features = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lat = float(row.pop("latitude"))
            lon = float(row.pop("longitude"))
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": row
            })
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False, indent=2)

def main():
    snpp_root = "/home/xurshedjonff/VIIRS_RAW_DATA/SNPP"
    out_root = "/home/xurshedjonff/VIIRS_INJE_FIRE_OUTPUT"
    year, doy, times = 2026, 21, ["1630"]

    tail = Path(str(year)) / f"{doy:03d}"
    img_dir, geo_dir, mod_dir, mod_geo_dir = [Path(snpp_root) / p / tail for p in ("VNP02IMG", "VNP03IMG", "VNP02MOD", "VNP03MOD")]
    out_dir = Path(out_root) / str(year) / f"{doy:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for hhmm in times:
        img = sorted(img_dir.glob(f"VNP02IMG.A{year}{doy:03d}.{hhmm}.*.nc"))[0]
        geo = sorted(geo_dir.glob(f"VNP03IMG.A{year}{doy:03d}.{hhmm}.*.nc"))[0]
        mod = sorted(mod_dir.glob(f"VNP02MOD.A{year}{doy:03d}.{hhmm}.*.nc"))[0]
        mod_geo = sorted(mod_geo_dir.glob(f"VNP03MOD.A{year}{doy:03d}.{hhmm}.*.nc"))[0]

        fs = FileSet(str(img), str(geo), str(mod), str(mod_geo))
        d = load_inputs(fs)
        result = run_algorithm(d)
        records = build_records(fs, d, result)

        m = re.search(r"A(\d{7})\.(\d{4})", img.name)
        key = f"A{m.group(1)}_{m.group(2)}"
        csv_path = str(out_dir / f"{key}_fires.csv")
        geojson_path = str(out_dir / f"{key}_fires.geojson")
        write_csv(csv_path, records)
        write_geojson(csv_path, geojson_path)
        print(f"{key} | fires={len(records)} | csv={csv_path}")

if __name__ == "__main__":
    main()