#!/usr/bin/env python3
"""
gk2a_frp.py
============

Research GK2A/AMI Fire Radiative Power (FRP) retrieval for event-based analysis.

IMPORTANT SCIENTIFIC SCOPE
--------------------------
This is a RESEARCH FRP retrieval, not a reproduction of an official NMSC FRP
product and not a reproduction of the GK2A FF detection contextual algorithm.

Primary FRP physics:
    Wooster MIR-radiance method

        FRP = A_pix * sigma / (a_AMI * tau_3p8) * (R_fire - R_bg)

This implementation uses:
  * official GK2A FF + DQF_FF to identify operational fire pixels,
  * official GK2A CLD to screen clear background pixels,
  * calibrated GK2A L1B SW038 radiance for FRP,
  * calibrated GK2A L1B IR112 for diagnostic BT and Delta-BT,
  * KFS event metadata from events.csv to locate the event ground-truth vector,
  * the KFS vector geometry to derive the GK2A KO crop window directly,
  * the verified GK2A KO Lambert grid definition to generate lat/lon directly.

Raw NetCDFs are read, cropped, calibrated, processed, and one compact pixel CSV is written directly.

Final CSV schema:
  * pixels: latitude, longitude, bt38, bt112, acq_date, acq_time, satellite,
    instrument, confidence, dqf_ff, frp, daynight, kfs_associated

For VIIRS/FIRMS compatibility, acq_date/acq_time are UTC. FRP is in MW and
BT values are in kelvin. Event-level time series can be derived later by grouping
this pixel CSV by acquisition time and summing FRP. By default, the CSV retains
official FF detections within 5 km of KFS, while kfs_associated=True keeps the
stricter 3-km primary matching criterion.

Current V1 assumptions:
  * tau_3p8 = 1.0 by default: TOA / no explicit atmospheric-transmittance correction.
  * research background = smallest adaptive clear, normal-land, non-fire window
    with enough valid pixels: 5x5 -> 7x7 -> 9x9 -> 11x11 -> 15x15 by default.
  * no raw-DN "hot outlier" rejection is used.
  * final KFS burned area is NOT treated as time-resolved active fire.
  * KFS is NOT excluded from background by default (optional sensitivity switch exists).
  * min_valid=8 is a provisional research parameter and should be sensitivity-tested.
  * daytime SW038 contains a reflected-solar component; local background subtraction
    mitigates but does not guarantee removal of all daytime reflectance bias.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import netCDF4 as nc
import numpy as np
from pyproj import CRS, Geod, Transformer

import download_gk2a as DL
import ff_matching as FM

try:
    import frame_quality as FQ
except Exception:
    FQ = None


# ---------------------------------------------------------------------
# Verified GK2A/AMI calibration constants
#
# Source:
# 20191115_gk-2a ami calibration table_v3.1_ir133_srf_shift.xlsx
# sheet: "coeff.& equation_WN"
#
# GK2A product/channel mapping:
#   SW038 -> workbook IR3.8
#   IR112 -> workbook IR11.2
#
# Native radiance unit:
#   mW m^-2 sr^-1 (cm^-1)^-1
# ---------------------------------------------------------------------

CAL = {
    "SW038": {
        "bits": 14,
        "wn_cm": 2612.67737352111,
        "gain": -0.00108296517282724,
        "offset": 17.699987411499,
        "c0": -0.447843939824124,
        "c1": 1.00065568090389,
        "c2": -6.33824089912448e-08,
    },
    "IR112": {
        "bits": 13,
        "wn_cm": 891.71305730126,
        "gain": -0.0216744858771562,
        "offset": 176.713439941406,
        "c0": -0.249111718496148,
        "c1": 1.00121166873756,
        "c2": -1.13167964011665e-06,
    },
}

# Workbook-provided Planck constants, retained for strict consistency with NMSC.
PLANCK_C1 = 1.1910428681415875e-16   # 2 h c^2
PLANCK_C2 = 0.014387769599838155     # h c / k

# Stefan-Boltzmann constant [W m^-2 K^-4].
SIGMA_SB = 5.670374419e-8

# Derived from the official 3388-point GK2A AMI SW038 SRF using Planck
# convolution over 650--1300 K at 1-K intervals and a zero-intercept
# least-squares fit L_native(T) ~= a_AMI_native T^4.
#
# Unit matches NMSC native calibrated SW038 radiance:
#   mW m^-2 sr^-1 (cm^-1)^-1 K^-4
A_AMI_NATIVE = 4.684109211472e-09

# Independent wavelength-domain derivation retained only as provenance/check:
#   3.195065676217e-09 W m^-2 sr^-1 um^-1 K^-4
A_AMI_WAVELENGTH = 3.195065676217e-09

# Verified SW038 SRF diagnostics:
SW038_SRF_WEIGHTED_WAVELENGTH_UM = 3.830302360714896
SW038_SRF_WEIGHTED_WAVENUMBER_CM = 2612.6773732097868

# The KO grid parameters verified to match FF / CLD / SW038 / IR112.
GRID_N = 900
GRID_PIXEL_M = 2000.0
GRID_UL_CENTER_X = -899000.0
GRID_UL_CENTER_Y = 899000.0

LCC_CRS = CRS.from_proj4(
    "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
WGS84 = CRS.from_epsg(4326)
TO_LL = Transformer.from_crs(LCC_CRS, WGS84, always_xy=True)
GEOD = Geod(ellps="WGS84")

FIRE_DQF_CODES = (7, 8, 9)
NORMAL_LAND_DQF = 2
CLEAR_CLD = 2

DEFAULT_WINDOWS = (5, 7, 9, 11, 15)
DEFAULT_CROP_BUFFER_KM = 20.0


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research GK2A/AMI SW038 FRP retrieval from event raw NetCDFs."
    )
    p.add_argument("--event", required=True, help="Event ID, e.g. 2023-04-03_Hampyeong")
    p.add_argument(
        "--start",
        default=None,
        help="Optional UTC start YYYYMMDDHHMM. Default: infer earliest SW038 file.",
    )
    p.add_argument(
        "--end",
        default=None,
        help="Optional UTC end YYYYMMDDHHMM. Default: infer latest SW038 file.",
    )
    p.add_argument("--step-min", type=int, default=10)
    p.add_argument(
        "--crop-buffer-km",
        type=float,
        default=DEFAULT_CROP_BUFFER_KM,
        help=(
            "KFS-bounds padding used to build the direct GK2A crop. Default: 20 km. "
            "The code automatically increases this if needed for the event buffer + "
            "largest background window."
        ),
    )
    p.add_argument(
        "--match-buffer-km",
        type=float,
        default=3.0,
        help=(
            "Primary KFS association tolerance in km. Pixels inside this tolerance "
            "are written with kfs_associated=True. Default: 3 km."
        ),
    )
    p.add_argument(
        "--event-buffer-km",
        type=float,
        default=5.0,
        help=(
            "Maximum KFS proximity for inclusion in the event FRP CSV. Official FF "
            "pixels between match-buffer-km and event-buffer-km are retained but "
            "written with kfs_associated=False. Distant pixels are excluded. Default: 5 km."
        ),
    )
    p.add_argument(
        "--windows",
        default="5,7,9,11,15",
        help="Adaptive square background windows, comma-separated odd integers.",
    )
    p.add_argument("--min-valid", type=int, default=8)
    p.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="SW038 atmospheric transmittance. V1 default 1.0 = no explicit correction.",
    )
    p.add_argument(
        "--exclude-kfs-bg",
        action="store_true",
        help=(
            "Sensitivity option: exclude final KFS burned-area cells from background. "
            "OFF by default to avoid future-truth leakage."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: DATA_ROOT/results/frp/<event>/",
    )
    return p.parse_args()


def parse_stamp(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M")


def fmt_stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def timeline(start: datetime, end: datetime, step_min: int) -> List[datetime]:
    if end < start:
        raise ValueError(f"end {end} precedes start {start}")
    out = []
    t = start
    step = timedelta(minutes=step_min)
    while t <= end:
        out.append(t)
        t += step
    return out


_TS_RE = re.compile(r"(\d{12})(?=\.nc$)")


def timestamps_in_dir(path: Path) -> List[datetime]:
    out = []
    for f in path.glob("*.nc"):
        m = _TS_RE.search(f.name)
        if m:
            out.append(parse_stamp(m.group(1)))
    return sorted(set(out))


def infer_time_bounds(raw_root: Path, start_arg: Optional[str], end_arg: Optional[str]) -> Tuple[datetime, datetime]:
    sw_dir = raw_root / "sw038"
    stamps = timestamps_in_dir(sw_dir)
    if not stamps:
        raise FileNotFoundError(f"No timestamped SW038 NetCDF files found in {sw_dir}")
    start = parse_stamp(start_arg) if start_arg else stamps[0]
    end = parse_stamp(end_arg) if end_arg else stamps[-1]
    return start, end


def find_product_file(raw_root: Path, product: str, dt: datetime) -> Optional[Path]:
    ts = fmt_stamp(dt)
    d = raw_root / product.lower()
    expected_names = {
        "ff": f"gk2a_FF_KO_{ts}.nc",
        "cld": f"gk2a_CLD_KO_{ts}.nc",
        "sw038": f"gk2a_SW038_KO_{ts}.nc",
        "ir112": f"gk2a_IR112_KO_{ts}.nc",
    }
    exact = d / expected_names[product.lower()]
    if exact.exists():
        return exact
    hits = sorted(d.glob(f"*{ts}.nc"))
    return hits[0] if hits else None


def _filled_array(x, fill, dtype):
    ma = np.ma.asarray(x)
    return np.asarray(np.ma.filled(ma, fill), dtype=dtype)


# ---------------------------------------------------------------------
# Direct KFS -> GK2A spatial template (NO hypercube/interim store)
# ---------------------------------------------------------------------

def load_event_record(event: str) -> Tuple[dict, Path]:
    """Read one event row directly from DATA_ROOT/events.csv."""
    events_csv = Path(getattr(DL, "EVENTS_CSV", DL.DATA_ROOT / "events.csv"))
    if not events_csv.exists():
        raise FileNotFoundError(f"events.csv not found: {events_csv}")

    matches = []
    with events_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("event_id") or "").strip() == event:
                matches.append(dict(row))

    if not matches:
        raise KeyError(f"event_id {event!r} not found in {events_csv}")
    if len(matches) > 1:
        raise ValueError(f"event_id {event!r} appears {len(matches)} times in {events_csv}")
    return matches[0], events_csv


def _derived_kfs_stem(event: str) -> str:
    """2021-01-06_Yeongdeok -> 2021_01_06_yeongdeok"""
    if "_" not in event:
        raise ValueError(f"Unexpected event_id format: {event!r}")
    date_part, place = event.split("_", 1)
    return f"{date_part.replace('-', '_')}_{place.lower()}"


def find_kfs_vector(event: str, ev: dict) -> Path:
    """
    Resolve the KFS vector from events.csv and the standardized
    raw/KFS_groundtruth_data directory.

    The derived event-id basename is tried as a fallback because some legacy rows
    in events.csv contain old .gpkg names while the standardized ground truth is .shp.
    """
    root = DL.DATA_ROOT / "raw" / "KFS_groundtruth_data"
    if not root.exists():
        raise FileNotFoundError(f"KFS ground-truth directory not found: {root}")

    candidates: List[Path] = []
    kfs_file = (ev.get("kfs_file") or "").strip()
    if kfs_file:
        p = Path(kfs_file)
        candidates.append(p if p.is_absolute() else root / p.name)
        candidates.extend([root / f"{p.stem}.shp", root / f"{p.stem}.gpkg"])

    stem = _derived_kfs_stem(event)
    candidates.extend([root / f"{stem}.shp", root / f"{stem}.gpkg"])

    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            return p

    # Last conservative fallback: same event date, then require a unique hit.
    date_prefix = event.split("_", 1)[0].replace("-", "_")
    hits = sorted(list(root.glob(f"{date_prefix}_*.shp")) + list(root.glob(f"{date_prefix}_*.gpkg")))
    if len(hits) == 1:
        return hits[0]

    tried = "\n  ".join(str(p) for p in candidates)
    extra = "\nDate candidates:\n  " + "\n  ".join(str(p) for p in hits) if hits else ""
    raise FileNotFoundError(
        f"Could not resolve KFS vector for {event}. Tried:\n  {tried}{extra}"
    )


def _make_valid(geom):
    """Shapely-version-tolerant make_valid helper."""
    try:
        from shapely import make_valid
        return make_valid(geom)
    except Exception:
        if geom is None:
            return geom
        return geom if geom.is_valid else geom.buffer(0)


def _kfs_burned_polygon(g5, target_km2: Optional[float] = None):
    """
    Build the burned-area polygon in EPSG:5179.

    Closed Polygon/MultiPolygon input is used directly. Line/perimeter-arc input
    follows the established project reconstruction logic: polygonize first, then
    buffered closure / convex-hull candidates, using events.csv area_ha as a
    target when available.
    """
    from shapely.ops import polygonize, unary_union

    geoms = [g for g in g5.geometry if g is not None and not g.is_empty]
    if not geoms:
        raise ValueError("KFS vector contains no non-empty geometry")

    types = {g.geom_type for g in geoms}
    if types <= {"Polygon", "MultiPolygon"}:
        return _make_valid(unary_union(geoms))

    lines = _make_valid(unary_union(geoms))
    faces = list(polygonize(lines))
    poly = _make_valid(unary_union(faces)) if faces else None
    poly_km2 = (poly.area / 1e6) if (poly is not None and not poly.is_empty) else 0.0

    candidates = []
    if poly is not None and not poly.is_empty:
        candidates.append(poly)

    for radius_m in (250.0, 500.0, 750.0, 1000.0):
        closed = lines.buffer(radius_m).buffer(-radius_m * 0.6)
        if not closed.is_empty:
            candidates.append(_make_valid(closed))

    if not lines.convex_hull.is_empty:
        candidates.append(lines.convex_hull)

    candidates = [g for g in candidates if g is not None and not g.is_empty and g.area > 0]
    if not candidates:
        burned = _make_valid(lines.convex_hull)
        if burned is None or burned.is_empty:
            raise ValueError("Could not reconstruct a burned polygon from KFS geometry")
        return burned

    if target_km2 and target_km2 > 0:
        if poly is not None and poly_km2 >= 0.40 * target_km2:
            return poly

        target_m2 = target_km2 * 1e6
        lo, hi, radius_m = 50.0, 5000.0, 1000.0
        for _ in range(14):
            radius_m = 0.5 * (lo + hi)
            test = lines.buffer(radius_m).buffer(-0.6 * radius_m)
            if test.area < target_m2:
                lo = radius_m
            else:
                hi = radius_m
        tuned = _make_valid(lines.buffer(radius_m).buffer(-0.6 * radius_m))
        if tuned is not None and not tuned.is_empty and tuned.area > 0:
            candidates.append(tuned)
        return min(candidates, key=lambda g: abs(g.area / 1e6 - target_km2))

    hull_km2 = lines.convex_hull.area / 1e6
    if poly is not None and poly_km2 >= 0.25 * hull_km2:
        return poly
    return max(candidates, key=lambda g: g.area)


def load_kfs_geometry(event: str):
    """Return (event_row, events_csv, kfs_path, burned_polygon_EPSG5179, target_km2)."""
    import geopandas as gpd

    ev, events_csv = load_event_record(event)
    kfs_path = find_kfs_vector(event, ev)
    gdf = gpd.read_file(kfs_path)
    if gdf.empty:
        raise ValueError(f"KFS vector is empty: {kfs_path}")
    if gdf.crs is None:
        print("WARNING: KFS vector has no CRS; assuming EPSG:5179:", kfs_path)
        gdf = gdf.set_crs(5179)
    g5 = gdf.to_crs(5179)

    area_text = (ev.get("area_ha") or "").strip()
    try:
        target_km2 = float(area_text) / 100.0 if area_text else None
    except ValueError:
        target_km2 = None

    burned = _kfs_burned_polygon(g5, target_km2=target_km2)
    if burned is None or burned.is_empty:
        raise ValueError(f"KFS burned polygon is empty for {event}")
    return ev, events_csv, kfs_path, burned, target_km2


def build_spatial_template_from_kfs(event: str, crop_buffer_km: float):
    """
    Derive [row_start,row_end,col_start,col_end] directly from the KFS burned-area
    bounds on the verified 900x900 GK2A KO Lambert grid, then generate the crop's
    latitude/longitude arrays from the same grid definition.
    """
    import geopandas as gpd

    if crop_buffer_km < 0:
        raise ValueError("--crop-buffer-km must be >= 0")

    ev, events_csv, kfs_path, burned_5179, target_km2 = load_kfs_geometry(event)
    burned_lcc = gpd.GeoSeries([burned_5179], crs=5179).to_crs(LCC_CRS).iloc[0]
    minx, miny, maxx, maxy = burned_lcc.bounds
    pad_m = float(crop_buffer_km) * 1000.0
    xmin, xmax = minx - pad_m, maxx + pad_m
    ymin, ymax = miny - pad_m, maxy + pad_m

    # Grid-cell centers:
    #   x(col) = GRID_UL_CENTER_X + col*GRID_PIXEL_M
    #   y(row) = GRID_UL_CENTER_Y - row*GRID_PIXEL_M
    c0 = math.floor((xmin - GRID_UL_CENTER_X) / GRID_PIXEL_M)
    c1 = math.ceil((xmax - GRID_UL_CENTER_X) / GRID_PIXEL_M) + 1
    r0 = math.floor((GRID_UL_CENTER_Y - ymax) / GRID_PIXEL_M)
    r1 = math.ceil((GRID_UL_CENTER_Y - ymin) / GRID_PIXEL_M) + 1

    c0 = max(0, min(GRID_N - 1, int(c0)))
    c1 = max(c0 + 1, min(GRID_N, int(c1)))
    r0 = max(0, min(GRID_N - 1, int(r0)))
    r1 = max(r0 + 1, min(GRID_N, int(r1)))

    cw = np.asarray([r0, r1, c0, c1], dtype=int)
    rows = np.arange(r0, r1, dtype=float)
    cols = np.arange(c0, c1, dtype=float)
    x = GRID_UL_CENTER_X + cols * GRID_PIXEL_M
    y = GRID_UL_CENTER_Y - rows * GRID_PIXEL_M
    xx, yy = np.meshgrid(x, y)
    lon, lat = TO_LL.transform(xx, yy)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    if lat.shape != (r1 - r0, c1 - c0) or lon.shape != lat.shape:
        raise RuntimeError(
            f"Generated lat/lon shape mismatch: crop={cw.tolist()}, "
            f"lat={lat.shape}, lon={lon.shape}"
        )

    return cw, lat, lon, ev, events_csv, kfs_path, burned_5179, target_km2


def rasterize_kfs_mask(
    event: str,
    burned_5179,
    lat: np.ndarray,
    lon: np.ndarray,
    target_km2: Optional[float] = None,
    cover_frac: float = 0.25,
) -> np.ndarray:
    """
    Rasterize the final KFS burned polygon onto the generated GK2A crop.

    This retains the established project rule: mark a cell when the burned polygon
    covers >=25% of the local ~2-km cell footprint. If a very small polygon marks no
    cell, assign its representative point to the nearest GK2A cell as a one-cell floor.
    """
    import geopandas as gpd
    from shapely.geometry import box as _box
    from shapely.strtree import STRtree

    H, W = lat.shape
    area4326 = gpd.GeoSeries([burned_5179], crs=5179).to_crs(4326).iloc[0]

    hlat = np.abs(np.gradient(lat, axis=0)) / 2.0
    hlon = np.abs(np.gradient(lon, axis=1)) / 2.0
    boxes = [
        _box(
            lon.flat[k] - hlon.flat[k],
            lat.flat[k] - hlat.flat[k],
            lon.flat[k] + hlon.flat[k],
            lat.flat[k] + hlat.flat[k],
        )
        for k in range(H * W)
    ]

    flat = np.zeros(H * W, dtype=bool)
    tree = STRtree(boxes)
    try:
        hits = np.asarray(tree.query(area4326, predicate="intersects"), dtype=int)
    except TypeError:
        # Compatibility fallback for older Shapely STRtree APIs.
        hit_geoms = tree.query(area4326)
        by_id = {id(g): i for i, g in enumerate(boxes)}
        hits = np.asarray([by_id[id(g)] for g in hit_geoms if id(g) in by_id], dtype=int)

    for k in hits:
        b = boxes[int(k)]
        if b.area > 0 and area4326.intersection(b).area >= float(cover_frac) * b.area:
            flat[int(k)] = True

    if not flat.any() and not area4326.is_empty:
        rp = area4326.representative_point()
        d2 = (lat - rp.y) ** 2 + (lon - rp.x) ** 2
        flat[int(np.nanargmin(d2))] = True

    mask = flat.reshape(H, W)
    burned_km2 = float(burned_5179.area / 1e6)
    tgt = f", target {target_km2:.2f} km2" if target_km2 else ""
    print(
        f"  [kfs] {event}: burned area {burned_km2:.2f} km2{tgt} -> "
        f"{int(mask.sum())} GK2A cells on {H}x{W} crop"
    )
    return mask


def validate_generated_lcc_grid(crop_window: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> float:
    """Cross-check one generated crop cell against the verified LCC transform."""
    r0, _, c0, _ = map(int, crop_window)
    lr, lc = lat.shape[0] // 2, lon.shape[1] // 2
    full_r = r0 + lr
    full_c = c0 + lc
    x = GRID_UL_CENTER_X + full_c * GRID_PIXEL_M
    y = GRID_UL_CENTER_Y - full_r * GRID_PIXEL_M
    calc_lon, calc_lat = TO_LL.transform(x, y)
    _, _, dist_m = GEOD.inv(calc_lon, calc_lat, float(lon[lr, lc]), float(lat[lr, lc]))
    if dist_m > 1.0:
        raise RuntimeError(f"Generated LCC grid self-check mismatch is {dist_m:.3f} m")
    return float(dist_m)


def solar_zenith_grid(lat: np.ndarray, lon: np.ndarray, stamps64: np.ndarray) -> np.ndarray:
    """Compute SZA directly through the project's geometry module; no interim store."""
    import sys
    code_parent = Path(__file__).resolve().parent.parent
    if str(code_parent) not in sys.path:
        sys.path.insert(0, str(code_parent))
    import geometry as G

    out = []
    for x in np.asarray(stamps64).ravel():
        dt = x.astype("datetime64[us]").item()
        z, _ = G.solar_position(lat, lon, dt)
        out.append(np.asarray(z, dtype=float))
    return np.stack(out)


def pixel_area_grid(crop_window: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Geodesic ground area [m^2] of each KO LCC 2-km projected cell.

    The full-grid pixel center is:
        x = -899000 + col*2000
        y = +899000 - row*2000

    Therefore cell edges are center +/- 1000 m.
    """
    r0, _, c0, _ = map(int, crop_window)
    H, W = shape
    out = np.full((H, W), np.nan, dtype=float)

    half = GRID_PIXEL_M / 2.0

    for r in range(H):
        full_r = r0 + r
        yc = GRID_UL_CENTER_Y - full_r * GRID_PIXEL_M
        y_top = yc + half
        y_bottom = yc - half

        for c in range(W):
            full_c = c0 + c
            xc = GRID_UL_CENTER_X + full_c * GRID_PIXEL_M
            x_left = xc - half
            x_right = xc + half

            xs = [x_left, x_right, x_right, x_left]
            ys = [y_top, y_top, y_bottom, y_bottom]
            lons, lats = TO_LL.transform(xs, ys)
            area, _ = GEOD.polygon_area_perimeter(lons, lats)
            out[r, c] = abs(area)

    return out


# ---------------------------------------------------------------------
# NetCDF reading
# ---------------------------------------------------------------------

def read_flag_crop(path: Path, varname: str, crop_window: np.ndarray) -> np.ndarray:
    r0, r1, c0, c1 = map(int, crop_window)
    with nc.Dataset(path) as ds:
        if varname not in ds.variables:
            raise KeyError(f"{varname} missing from {path}")
        x = ds.variables[varname][r0:r1, c0:c1]
    return _filled_array(x, 255, np.uint8)


def read_l1b_crop(path: Path, bits: int, crop_window: np.ndarray):
    """
    Read packed uint16 L1B values and preserve validity BEFORE bit masking.

    Returns
    -------
    raw   : uint16 [H,W]
    dn    : uint16 [H,W], valid low-bit count extracted
    valid : bool   [H,W], based on the original NetCDF mask/fill state

    The explicit raw != 65535 test prevents a uint16 fill value from becoming
    apparently-valid 16383/8191 after bit masking.
    """
    r0, r1, c0, c1 = map(int, crop_window)
    with nc.Dataset(path) as ds:
        if "image_pixel_values" not in ds.variables:
            raise KeyError(f"image_pixel_values missing from {path}")
        x = np.ma.asarray(ds.variables["image_pixel_values"][r0:r1, c0:c1])
        mask = np.ma.getmaskarray(x)
        raw = np.asarray(np.ma.filled(x, 65535), dtype=np.uint16)

    valid = (~mask) & (raw != np.uint16(65535))
    bitmask = np.uint16((1 << bits) - 1)
    dn = raw & bitmask
    return raw, dn, valid


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

def dn_to_native_radiance(dn: np.ndarray, valid: np.ndarray, cfg: dict):
    rad = cfg["gain"] * dn.astype(np.float64) + cfg["offset"]
    valid2 = np.asarray(valid, bool) & np.isfinite(rad) & (rad > 0.0)
    rad = np.where(valid2, rad, np.nan)
    return rad, valid2


def native_radiance_to_bt(rad: np.ndarray, cfg: dict) -> np.ndarray:
    """
    NMSC v3.1 radiance -> effective temperature -> brightness temperature.

    rad unit:
        mW m^-2 sr^-1 (cm^-1)^-1
    """
    rad = np.asarray(rad, dtype=float)
    nu_m = cfg["wn_cm"] * 100.0

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        te = (
            PLANCK_C2 * nu_m
            / np.log(PLANCK_C1 * nu_m**3 / (rad * 1e-5) + 1.0)
        )
        bt = cfg["c0"] + cfg["c1"] * te + cfg["c2"] * te**2

    bt[~np.isfinite(rad) | (rad <= 0)] = np.nan
    return bt


# ---------------------------------------------------------------------
# Research FRP background
# ---------------------------------------------------------------------

def _window_slice(H: int, W: int, row: int, col: int, half: int):
    r0 = max(0, row - half)
    r1 = min(H, row + half + 1)
    c0 = max(0, col - half)
    c1 = min(W, col + half + 1)
    return r0, r1, c0, c1


def local_background_native_radiance(
    rad38: np.ndarray,
    valid38: np.ndarray,
    cld: np.ndarray,
    ff: np.ndarray,
    dqf: np.ndarray,
    row: int,
    col: int,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    min_valid: int = 8,
    kfs_mask: Optional[np.ndarray] = None,
    exclude_kfs: bool = False,
):
    """
    Estimate local non-fire SW038 background radiance for one target pixel.

    Candidate background pixel must be:
      * valid calibrated SW038 radiance,
      * CLD == 2 (confident clear),
      * DQF_FF == 2 (normal land),
      * FF == 0,
      * not the target pixel,
      * optionally outside final KFS burned-area mask.

    No thermal/DN outlier filter is applied in V1.
    Returns (median_radiance, selected_window, n_valid).
    """
    H, W = rad38.shape
    if not (0 <= row < H and 0 <= col < W):
        raise IndexError(f"row,col=({row},{col}) outside field shape {(H,W)}")

    for w in windows:
        if w < 3 or w % 2 == 0:
            raise ValueError(f"Background windows must be odd integers >=3, got {w}")
        half = w // 2
        r0, r1, c0, c1 = _window_slice(H, W, row, col, half)

        sub_r = rad38[r0:r1, c0:c1]
        valid = (
            np.isfinite(sub_r)
            & valid38[r0:r1, c0:c1]
            & (cld[r0:r1, c0:c1] == CLEAR_CLD)
            & (dqf[r0:r1, c0:c1] == NORMAL_LAND_DQF)
            & (ff[r0:r1, c0:c1] == 0)
        )

        if exclude_kfs:
            if kfs_mask is None:
                raise ValueError("exclude_kfs=True requires kfs_mask")
            valid &= ~kfs_mask[r0:r1, c0:c1]

        rr = row - r0
        cc = col - c0
        if 0 <= rr < valid.shape[0] and 0 <= cc < valid.shape[1]:
            valid[rr, cc] = False

        n = int(valid.sum())
        if n >= min_valid:
            return float(np.median(sub_r[valid])), int(w), n

    return float("nan"), None, 0


# ---------------------------------------------------------------------
# FRP
# ---------------------------------------------------------------------

def frp_mw_from_native_radiance(
    fire_rad: float,
    bg_rad: float,
    pixel_area_m2: float,
    tau: float,
) -> float:
    """
    Wooster-style MIR-radiance FRP in MW using native NMSC radiance units.

    a_AMI_native was fitted in exactly the same native radiance units as fire_rad
    and bg_rad, so their mW spectral-density unit cancels in the radiance/a ratio.
    """
    if not (0 < tau <= 1.0):
        raise ValueError(f"tau must satisfy 0 < tau <= 1, got {tau}")
    if not (np.isfinite(fire_rad) and np.isfinite(bg_rad) and np.isfinite(pixel_area_m2)):
        return float("nan")

    delta = fire_rad - bg_rad
    if delta <= 0:
        return float("nan")

    frp_w = pixel_area_m2 * SIGMA_SB / (A_AMI_NATIVE * tau) * delta
    return float(frp_w / 1e6)


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]):
    """Write only requested columns, atomically, even when rows contain diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    tmp.replace(path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    windows = tuple(int(x.strip()) for x in args.windows.split(",") if x.strip())
    if not windows:
        raise ValueError("No background windows supplied")
    if any(w < 3 or w % 2 == 0 for w in windows):
        raise ValueError(f"All windows must be odd integers >=3: {windows}")
    if args.min_valid < 1:
        raise ValueError("--min-valid must be >=1")
    if args.crop_buffer_km < 0:
        raise ValueError("--crop-buffer-km must be >=0")
    if args.match_buffer_km < 0:
        raise ValueError("--match-buffer-km must be >=0")
    if args.event_buffer_km < args.match_buffer_km:
        raise ValueError("--event-buffer-km must be >= --match-buffer-km")
    if not (0 < args.tau <= 1.0):
        raise ValueError("--tau must satisfy 0 < tau <= 1")

    event = args.event
    raw_root = DL.DATA_ROOT / "raw" / event / "gk2a"

    # The crop must include both the event-association tolerance and enough halo
    # for the largest adaptive background window around those event pixels.
    half_bg_km = (max(windows) // 2) * (GRID_PIXEL_M / 1000.0)
    min_safe_crop_km = args.event_buffer_km + half_bg_km
    crop_buffer_km = max(float(args.crop_buffer_km), float(min_safe_crop_km))
    if crop_buffer_km > args.crop_buffer_km:
        print(
            f"NOTE: increasing crop buffer from {args.crop_buffer_km:g} to "
            f"{crop_buffer_km:g} km to preserve event-buffer + background context."
        )

    (
        crop_window, lat, lon, ev, events_csv, kfs_path, burned_5179, target_km2
    ) = build_spatial_template_from_kfs(event, crop_buffer_km)
    H, W = lat.shape

    grid_mismatch_m = validate_generated_lcc_grid(crop_window, lat, lon)
    area_m2 = pixel_area_grid(crop_window, (H, W))

    start, end = infer_time_bounds(raw_root, args.start, args.end)
    stamps = timeline(start, end, args.step_min)
    stamps64 = np.asarray(stamps, dtype="datetime64[m]")

    # Static final KFS burned-area mask for event association / validation.
    # No hypercube or interim store is read.
    kfs_mask = rasterize_kfs_mask(
        event, burned_5179, lat, lon, target_km2=target_km2, cover_frac=0.25
    )


    print("\n=== GK2A FRP CONFIGURATION ===")
    print("event:", event)
    print("events.csv:", events_csv)
    print("KFS vector:", kfs_path)
    print("raw root:", raw_root)
    print("timeline:", len(stamps), "scans", start, "->", end, "UTC")
    print("crop source: KFS bounds +", f"{crop_buffer_km:g} km buffer")
    print("crop_window:", crop_window.tolist(), "shape:", (H, W))
    print("LCC generated-grid check:", f"{grid_mismatch_m:.3f} m")
    print(
        "pixel area km2: min/mean/max =",
        f"{np.nanmin(area_m2)/1e6:.6f}",
        f"{np.nanmean(area_m2)/1e6:.6f}",
        f"{np.nanmax(area_m2)/1e6:.6f}",
    )
    print("KFS cells:", int(kfs_mask.sum()))
    print("background windows:", windows, "min_valid:", args.min_valid)
    print("primary KFS association buffer:", args.match_buffer_km, "km")
    print("event export proximity buffer:", args.event_buffer_km, "km")
    print("tau_3.8:", args.tau)
    print("A_AMI_native:", f"{A_AMI_NATIVE:.12e}")
    print("NOTE: research FRP; no explicit atmospheric correction when tau=1.")

    # Optional geometry diagnostic. Failure does not block FRP.
    sza = None
    try:
        sza = solar_zenith_grid(lat, lon, stamps64)
        if sza.shape != (len(stamps), H, W):
            print("WARNING: unexpected SZA shape", sza.shape, "- disabling SZA output")
            sza = None
    except Exception as e:
        print("WARNING: could not compute SZA; continuing without it:", e)

    frames: List[Optional[dict]] = []
    missing_by_time: List[Tuple[datetime, List[str]]] = []

    for dt in stamps:
        paths = {
            "ff": find_product_file(raw_root, "ff", dt),
            "cld": find_product_file(raw_root, "cld", dt),
            "sw038": find_product_file(raw_root, "sw038", dt),
            "ir112": find_product_file(raw_root, "ir112", dt),
        }
        missing = [k for k, pth in paths.items() if pth is None]
        if missing:
            frames.append(None)
            missing_by_time.append((dt, missing))
            continue

        ff = read_flag_crop(paths["ff"], "FF", crop_window)
        dqf = read_flag_crop(paths["ff"], "DQF_FF", crop_window)
        cld = read_flag_crop(paths["cld"], "CLD", crop_window)

        sw_raw, sw_dn, sw_valid_raw = read_l1b_crop(
            paths["sw038"], CAL["SW038"]["bits"], crop_window
        )
        ir_raw, ir_dn, ir_valid_raw = read_l1b_crop(
            paths["ir112"], CAL["IR112"]["bits"], crop_window
        )

        rad38, valid38 = dn_to_native_radiance(sw_dn, sw_valid_raw, CAL["SW038"])
        rad112, valid112 = dn_to_native_radiance(ir_dn, ir_valid_raw, CAL["IR112"])
        bt38 = native_radiance_to_bt(rad38, CAL["SW038"])
        bt112 = native_radiance_to_bt(rad112, CAL["IR112"])
        delta_bt = bt38 - bt112

        frames.append(
            {
                "ff": ff,
                "dqf": dqf,
                "cld": cld,
                "sw_raw": sw_raw,
                "sw_dn": sw_dn,
                "sw_valid_raw": sw_valid_raw,
                "ir_raw": ir_raw,
                "ir_dn": ir_dn,
                "ir_valid_raw": ir_valid_raw,
                "rad38": rad38,
                "rad112": rad112,
                "valid38": valid38,
                "valid112": valid112,
                "bt38": bt38,
                "bt112": bt112,
                "delta_bt": delta_bt,
            }
        )

    # Frame-artifact flag from the project's existing helper. This is reported,
    # not automatically dropped from FRP.
    artifact_flags = np.zeros(len(stamps), dtype=bool)
    if FQ is not None:
        try:
            ff_cube = np.zeros((len(stamps), H, W), dtype=bool)
            for i, fr in enumerate(frames):
                if fr is not None:
                    ff_cube[i] = (np.asarray(fr["ff"]) == 1)
            artifact_flags = np.asarray(FQ.artifact_frames(ff_cube, kfs_mask), dtype=bool)
        except Exception as e:
            print("WARNING: frame_quality.artifact_frames failed:", e)

    print("\n=== AVAILABILITY ===")
    print("nominal scans:", len(stamps))
    print("complete scans:", sum(fr is not None for fr in frames))
    print("missing scans:", len(missing_by_time))
    for dt, miss in missing_by_time:
        print("  missing", dt.strftime("%Y-%m-%d %H:%M"), ",".join(miss))
    if artifact_flags.any():
        print("artifact-flagged frames:", int(artifact_flags.sum()))
    else:
        print("artifact-flagged frames: 0")

    pixel_rows: List[dict] = []
    n_official_ff = 0
    n_primary_assoc = 0
    n_near_extra = 0
    n_far_excluded = 0

    for ti, dt in enumerate(stamps):
        fr = frames[ti]
        if fr is None:
            continue

        ff = fr["ff"]
        dqf = fr["dqf"]
        cld = fr["cld"]

        # Official operational forest-fire pixels only.
        fire_mask = (ff == 1) & np.isin(dqf, FIRE_DQF_CODES)

        # Two KFS proximity roles are intentionally kept separate:
        #   1) match-buffer-km (default 3 km): primary validation association.
        #      Pixels inside this tolerance get kfs_associated=True.
        #   2) event-buffer-km (default 5 km): event-export limit.
        #      Nearby official FF pixels just outside the primary tolerance are
        #      restored to the CSV, while distant/unrelated detections are excluded.
        match_primary = FM.match_at_buffer(
            fire_mask,
            kfs_mask,
            buffer_km=args.match_buffer_km,
        )
        associated = np.asarray(match_primary["tp_mask"], dtype=bool)

        match_event = FM.match_at_buffer(
            fire_mask,
            kfs_mask,
            buffer_km=args.event_buffer_km,
        )
        event_fire_mask = np.asarray(match_event["tp_mask"], dtype=bool)

        near_extra_mask = event_fire_mask & ~associated
        far_excluded_mask = fire_mask & ~event_fire_mask

        n_official_ff += int(fire_mask.sum())
        n_primary_assoc += int(associated.sum())
        n_near_extra += int(near_extra_mask.sum())
        n_far_excluded += int(far_excluded_mask.sum())

        ys, xs = np.where(event_fire_mask)

        for r, c in zip(ys.tolist(), xs.tolist()):
            bg_rad, _, _ = local_background_native_radiance(
                fr["rad38"],
                fr["valid38"],
                cld,
                ff,
                dqf,
                r,
                c,
                windows=windows,
                min_valid=args.min_valid,
                kfs_mask=kfs_mask,
                exclude_kfs=args.exclude_kfs_bg,
            )

            fire_rad = float(fr["rad38"][r, c])
            frp_mw = frp_mw_from_native_radiance(
                fire_rad=fire_rad,
                bg_rad=bg_rad,
                pixel_area_m2=float(area_m2[r, c]),
                tau=args.tau,
            )


            sza_val = (
                float(sza[ti, r, c])
                if sza is not None and np.isfinite(sza[ti, r, c])
                else float("nan")
            )
            regime = (
                "day" if np.isfinite(sza_val) and sza_val < 85.0
                else "night" if np.isfinite(sza_val)
                else ""
            )

            # Compact FIRMS/VIIRS-style scientific output.
            # acq_date/acq_time are UTC, matching the convention used by FIRMS.
            # Retain official FF detections close to KFS (default <=5 km).
            # kfs_associated remains the stricter primary match (default <=3 km).
            # FRP is NaN if retrieval failed.
            confidence = {7: "low", 8: "medium", 9: "high"}.get(int(dqf[r, c]), "")
            daynight = "D" if regime == "day" else "N" if regime == "night" else ""

            pixel_rows.append(
                {
                    "latitude": float(lat[r, c]),
                    "longitude": float(lon[r, c]),
                    "bt38": float(fr["bt38"][r, c]),
                    "bt112": float(fr["bt112"][r, c]),
                    "acq_date": dt.strftime("%Y-%m-%d"),
                    "acq_time": int(dt.strftime("%H%M")),
                    "satellite": "GK2A",
                    "instrument": "AMI",
                    "confidence": confidence,
                    "dqf_ff": int(dqf[r, c]),
                    "frp": frp_mw,
                    "daynight": daynight,
                    "kfs_associated": bool(associated[r, c]),
                }
            )


    if args.output_dir:
        outdir = Path(args.output_dir)
    else:
        outdir = DL.DATA_ROOT / "results" / "frp" / event
    outdir.mkdir(parents=True, exist_ok=True)

    px_path = outdir / f"{event}_gk2a_frp_pixels.csv"

    # The pixel CSV is the only persistent FRP output.
    pixel_fields = [
        "latitude",
        "longitude",
        "bt38",
        "bt112",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
        "confidence",
        "dqf_ff",
        "frp",
        "daynight",
        "kfs_associated",
    ]

    write_csv(px_path, pixel_rows, pixel_fields)

    print("\n=== EVENT PROXIMITY FILTER ===")
    print("official FF detections in crop:", n_official_ff)
    print(f"primary KFS-associated (<= {args.match_buffer_km:g} km):", n_primary_assoc)
    print(
        f"near-KFS detections restored ({args.match_buffer_km:g} to "
        f"{args.event_buffer_km:g} km):",
        n_near_extra,
    )
    print(f"distant detections excluded (> {args.event_buffer_km:g} km):", n_far_excluded)

    print("\n=== OUTPUT ===")
    print("pixels:", px_path)
    print("pixel rows:", len(pixel_rows))

    ok_px = [r["frp"] for r in pixel_rows if np.isfinite(float(r["frp"]))]
    print("finite pixel FRPs:", len(ok_px))
    if ok_px:
        print(
            "pixel FRP MW min/median/max:",
            f"{np.nanmin(ok_px):.6f}",
            f"{np.nanmedian(ok_px):.6f}",
            f"{np.nanmax(ok_px):.6f}",
        )

    print("\nMETHOD NOTE:")
    print("  Research GK2A/AMI FRP using official SW038 calibration and")
    print("  an AMI-SRF-derived Wooster MIR coefficient.")
    print("  tau=1 means TOA/no explicit atmospheric correction.")
    print("  KFS is used for crop construction + event association, not as time-resolved active-fire area.")
    print("  Spatial template is generated directly from events.csv + KFS; no hypercube/interim store is used.")
    print(
        f"  Primary KFS association = {args.match_buffer_km:g} km; event export proximity = "
        f"{args.event_buffer_km:g} km."
    )
    print("  Background window/min_valid choices remain sensitivity-test parameters.")


if __name__ == "__main__":
    main()