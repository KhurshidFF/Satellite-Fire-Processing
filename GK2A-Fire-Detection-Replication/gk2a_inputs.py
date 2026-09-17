# -*- coding: utf-8 -*-
"""
gk2a_inputs.py
==============
Helper module that builds the *missing* inputs for the GK2A FF replication:

    1. Real Solar Zenith Angle (SZA)        from scan time + lat/lon
    2. Real View  Zenith Angle (VZA)        from satellite geometry
    3. Baresoil / urban / cropland mask     from a landcover GeoTIFF
    4. Industrial heat-source mask          from ATBD Table 2.7 site list
    5. Topographic reference pixel indices  from local DEM variability

These are exactly the inputs your current run script was passing as None or as
hardcoded placeholders, which caused the false alarms outside the burn area.

Dependencies
------------
Required: numpy, scipy
Optional: rasterio  (only if you use a landcover GeoTIFF)

Typical use
-----------
    import numpy as np
    import netCDF4 as nc
    from datetime import datetime
    from gk2a_inputs import (
        compute_sza, compute_vza,
        build_baresoil_mask_from_geotiff,
        build_industrial_mask,
        pick_topo_reference_pixels,
    )
    from gk2a_ff_algorithm import run_ff_algorithm, summarise

    # ... load LAT, LON, T07, T14, R03, R04, LAND, CLOUD, DEM as before ...

    scan_dt = datetime(2026, 2, 21, 19, 30)         # UTC of the GK2A scan

    SZA      = compute_sza(LAT, LON, scan_dt)
    VZA      = compute_vza(LAT, LON, sub_lon=128.2)
    IND_MASK = build_industrial_mask(LAT, LON)
    BARESOIL = build_baresoil_mask_from_geotiff(
                   "/path/to/ESA_WorldCover_2021_korea.tif", LAT, LON,
                   classes=(40, 50, 60))             # cropland + built-up + bare
    R, C     = pick_topo_reference_pixels(DEM, LAND, CLOUD, n_pixels=700)

    result = run_ff_algorithm(
        T_SWIR=T07, T_TIR=T14, refl_red=R03, refl_nir=R04,
        sza=SZA, vza=VZA,
        land_mask=LAND, cloud_mask=CLOUD, dem=DEM,
        industrial_mask=IND_MASK,
        baresoil_mask=BARESOIL,
        topo_ref_rows=R, topo_ref_cols=C,
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, Sequence, Tuple, List
import warnings

import numpy as np
from scipy.ndimage import generic_filter


# =============================================================================
# 1. Solar Zenith Angle
# =============================================================================

def compute_sza(lat, lon, dt_utc: datetime) -> np.ndarray:
    """
    Compute Solar Zenith Angle [degrees] for every pixel of a satellite grid.

    Uses the standard solar position approximation (NOAA solar formula):
        - declination delta = 23.45 * sin(360/365 * (DOY - 81))
        - hour angle H      = 15 * (LST - 12)        in degrees
        - cos(SZA) = sin(lat)*sin(delta) + cos(lat)*cos(delta)*cos(H)

    Accuracy: about +-0.5 deg, far better than the 90 deg placeholder.
    For sub-degree accuracy use astropy.coordinates.get_sun(), but this
    formula is sufficient for the SZA < 85 deg day/night cutoff used by
    the GK2A FF algorithm.

    Parameters
    ----------
    lat, lon  : (H, W) float arrays of pixel latitude / longitude [deg]
    dt_utc    : datetime object giving the UTC time of the satellite scan
                (do NOT pass a local KST time)

    Returns
    -------
    sza : (H, W) float32 array of SZA values in degrees, range [0, 180]
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Day of year and decimal hour in UTC
    doy = dt_utc.timetuple().tm_yday
    hour_utc = (dt_utc.hour + dt_utc.minute / 60.0 +
                dt_utc.second / 3600.0)

    # Solar declination (degrees)
    decl_deg = 23.45 * np.sin(np.deg2rad(360.0 / 365.0 * (doy - 81)))
    decl     = np.deg2rad(decl_deg)

    # Equation of time correction (minutes) - small but worth including
    B = np.deg2rad(360.0 / 365.0 * (doy - 81))
    eot_min = 9.87 * np.sin(2*B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)

    # Local solar time (hours) at each pixel
    lst = hour_utc + lon / 15.0 + eot_min / 60.0

    # Hour angle (radians)
    H = np.deg2rad(15.0 * (lst - 12.0))

    lat_r = np.deg2rad(lat)

    cos_sza = (np.sin(lat_r) * np.sin(decl) +
               np.cos(lat_r) * np.cos(decl) * np.cos(H))
    cos_sza = np.clip(cos_sza, -1.0, 1.0)
    sza = np.rad2deg(np.arccos(cos_sza))

    return sza.astype(np.float32)


# =============================================================================
# 2. View Zenith Angle
# =============================================================================

def compute_vza(lat, lon, sub_lon: float = 128.2,
                sub_lat: float = 0.0) -> np.ndarray:
    """
    Compute View Zenith Angle [degrees] for a geostationary satellite.

    Geometry:
      - Pixel-to-sub-satellite great-circle angle:
            cos(a) = sin(sub_lat)*sin(lat)
                   + cos(sub_lat)*cos(lat)*cos(lon - sub_lon)
      - Slant distance from satellite to pixel:
            d = sqrt(H_sat^2 + Re^2 - 2*H_sat*Re*cos(a))
      - VZA at the pixel from local zenith to satellite (sine rule):
            sin(VZA) = H_sat * sin(a) / d

    For GK2A (sub_lon=128.2 deg E, geostationary), this gives ~40 deg VZA
    over central Korea, ~38 deg over Jeju, and >70 deg only beyond the
    visible limb of the Earth disk.

    Parameters
    ----------
    lat, lon : (H, W) lat/lon arrays [deg]
    sub_lon  : sub-satellite longitude [deg]; default 128.2 for GK2A
    sub_lat  : sub-satellite latitude  [deg]; 0.0 for geostationary orbits

    Returns
    -------
    vza : (H, W) float32 array, range [0, 90].
          Pixels beyond the Earth's visible limb are returned as 90.0.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    Re = 6378.137      # Earth equatorial radius [km]
    H_sat = 42164.0    # geostationary orbit radius from Earth center [km]

    lat_r     = np.deg2rad(lat)
    sub_lat_r = np.deg2rad(sub_lat)
    dlon      = np.deg2rad(lon - sub_lon)

    # Great-circle angle from sub-satellite point to pixel
    cos_a = (np.sin(sub_lat_r) * np.sin(lat_r) +
             np.cos(sub_lat_r) * np.cos(lat_r) * np.cos(dlon))
    cos_a = np.clip(cos_a, -1.0, 1.0)
    sin_a = np.sqrt(1.0 - cos_a ** 2)

    # Slant range from satellite to pixel
    d = np.sqrt(H_sat ** 2 + Re ** 2 - 2.0 * H_sat * Re * cos_a)

    # Sine rule for the angle at the pixel from local zenith to satellite
    sin_vza = H_sat * sin_a / d
    sin_vza = np.clip(sin_vza, 0.0, 1.0)
    vza = np.rad2deg(np.arcsin(sin_vza))

    # Pixels beyond the Earth's visible limb (cos_a < Re/H_sat) are not
    # actually viewable by the satellite -> set to 90.0
    vza = np.where(cos_a < Re / H_sat, 90.0, vza)

    return vza.astype(np.float32)


# =============================================================================
# 3. Baresoil / urban mask from a landcover GeoTIFF
# =============================================================================

def build_baresoil_mask_from_geotiff(
    geotiff_path: str,
    lat: np.ndarray,
    lon: np.ndarray,
    classes: Sequence[int] = (40, 50, 60),
) -> np.ndarray:
    """
    Resample a landcover GeoTIFF onto the satellite lat/lon grid and produce a
    boolean mask of pixels belonging to one of `classes`.

    Recommended landcover products (free):

        ESA WorldCover 2021 (10 m, global)
            URL: https://esa-worldcover.org
            Class codes you typically want to mask:
                40 = cropland
                50 = built-up / urban
                60 = bare / sparse vegetation

        Copernicus Global Land Cover 100 m
            URL: https://lcviewer.vito.be
            Class codes:
                40 = cropland
                50 = urban
                60 = bare

        MODIS MCD12Q1 IGBP scheme (500 m, yearly)
            Class codes:
                12 = cropland
                13 = urban
                16 = barren

    For each satellite pixel we sample the *nearest* landcover pixel.
    For coarse satellite grids (2 km GK2A) and a fine landcover input you
    will systematically miss small clearings, but this is fine for the
    purpose of rejecting whole valley pixels that read as warm.

    Parameters
    ----------
    geotiff_path : path to a GeoTIFF in EPSG:4326 (lat/lon WGS84)
    lat, lon     : (H, W) GK2A grid coordinates [deg]
    classes      : iterable of integer class codes to mark as True

    Returns
    -------
    mask : (H, W) bool array, True where the landcover class is in `classes`
    """
    try:
        import rasterio
    except ImportError as e:
        raise ImportError(
            "rasterio is required for landcover resampling.\n"
            "Install with:  pip install rasterio\n"
            "Or build the baresoil mask externally and load it with np.load()."
        ) from e

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    H, W = lat.shape

    with rasterio.open(geotiff_path) as src:
        # Sample the GeoTIFF at every (lon, lat) point of the satellite grid
        # rasterio expects (x, y) = (lon, lat)
        coords = list(zip(lon.ravel().tolist(), lat.ravel().tolist()))
        # src.sample yields one tuple per coordinate
        samples = np.fromiter(
            (s[0] for s in src.sample(coords)),
            dtype=np.int32,
            count=H * W,
        )
        lc_grid = samples.reshape(H, W)

    mask = np.isin(lc_grid, list(classes))
    return mask


def build_baresoil_mask_from_array(
    landcover_grid: np.ndarray,
    classes: Sequence[int] = (40, 50, 60),
) -> np.ndarray:
    """
    If you have already resampled a landcover array onto your GK2A grid
    (same shape as T_SWIR), this just produces the boolean mask.
    """
    return np.isin(np.asarray(landcover_grid), list(classes))


# -----------------------------------------------------------------------------
# Korean Ministry of Environment Land Cover (sub-class) class codes
# -----------------------------------------------------------------------------
# Reference: Korean Ministry of Environment Level-2 Subclass Land Cover Map
# (the "se-bun-ryu to-ji-pi-bok-ji-do" / sebunryu landcover map).
# Distributed via the ME Environmental Geographic Information Service (EGIS).
# Class codes follow the standard L2_CODE field of the official GPKG/SHP files.

KOR_LC_CLASSES = {
    # --- Urban / built-up (reject for fire) ---
    110: "Residential",
    120: "Industrial",
    130: "Commercial",
    140: "Cultural/sports/recreation",
    150: "Transportation",
    160: "Public facilities",
    # --- Cropland (reject for fire - bright, often warm in winter) ---
    210: "Rice paddy",
    220: "Dry field (upland crop)",
    230: "Greenhouse / facility farming",
    240: "Orchard",
    250: "Other cultivated land",
    # --- Forest (do NOT reject - real fires happen here) ---
    310: "Broadleaf forest",
    320: "Conifer forest",
    330: "Mixed forest",
    # --- Grassland ---
    410: "Natural grassland",
    420: "Artificial grassland",
    # --- Wetland ---
    510: "Inland wetland",
    520: "Coastal wetland",
    # --- Bare ground (reject for fire) ---
    610: "Natural bare land",
    620: "Artificial bare land",
    # --- Water (already handled by water mask) ---
    710: "Inland water",
    720: "Sea water",
}

# Default rejection classes for the Korean MoE landcover dataset.
# Includes urban, cropland (excluding orchards), and bare ground.
# Forests, grasslands, and wetlands remain eligible for fire detection.
KOR_LC_REJECT_DEFAULT = (
    110, 120, 130, 140, 150, 160,   # urban / built-up
    210, 220, 230, 250,              # cropland (240 orchards kept eligible)
    610, 620,                        # bare ground
)


def build_baresoil_mask_from_gpkg(
    gpkg_path: str,
    lat: np.ndarray,
    lon: np.ndarray,
    classes: Sequence[int] = KOR_LC_REJECT_DEFAULT,
    code_field: str = "L2_CODE",
    layer: Optional[str] = None,
    coverage_threshold: float = 0.50,
    bbox_padding_deg: float = 0.05,
    fine_res_deg: float = 0.005,
) -> np.ndarray:
    """
    Build a baresoil/urban/cropland rejection mask from a Korean Ministry of
    Environment Level-2 Subclass Land Cover GPKG (or any GPKG/SHP with an
    integer class code field).

    The GK2A 2 km pixel covers ~4 km^2 and contains thousands of small
    landcover polygons. We rasterize the GPKG to a fine grid, then aggregate
    to the GK2A grid using the **fraction of rejection-class area** inside
    each satellite pixel. A pixel is masked when this fraction exceeds
    `coverage_threshold` (default 50 %).

    Implementation note
    -------------------
    We do NOT pass a bbox to read_file. The Korean MoE GPKG is stored in
    EPSG:5186 (projected metres), and several geopandas + pyogrio + fiona
    version combinations silently mishandle the lat/lon -> projected CRS
    bbox conversion and return 0 polygons. Reading without a bbox is slower
    on the first call but correct, and 9 M polygons fit comfortably in RAM.

    Parameters
    ----------
    gpkg_path        : path to the .gpkg or .shp file
    lat, lon         : (H, W) GK2A grid coordinates [deg]
    classes          : iterable of class code integers to reject
    code_field       : attribute name of the integer class code column
                       (default 'L2_CODE')
    layer            : layer name inside the GPKG; None = first layer
    coverage_threshold : pixel-area fraction required to mask (0.0-1.0)
    bbox_padding_deg : padding around the satellite domain when clipping
    fine_res_deg     : intermediate raster resolution. 0.005 deg ~ 555 m,
                       so each 2 km GK2A pixel contains ~16 fine cells.

    Returns
    -------
    mask : (H, W) bool array
    """
    try:
        import geopandas as gpd
        from rasterio import features
        from rasterio.transform import from_origin
    except ImportError as e:
        raise ImportError(
            "build_baresoil_mask_from_gpkg requires geopandas + rasterio:\n"
            "    pip install geopandas rasterio shapely\n"
        ) from e

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.shape != lon.shape:
        raise ValueError("lat and lon must have identical shapes")
    H, W = lat.shape

    lon_min = float(np.nanmin(lon)) - bbox_padding_deg
    lon_max = float(np.nanmax(lon)) + bbox_padding_deg
    lat_min = float(np.nanmin(lat)) - bbox_padding_deg
    lat_max = float(np.nanmax(lat)) + bbox_padding_deg

    print(f"[gk2a_inputs] Reading GPKG (no bbox filter; ~30-90 s for the "
          f"full Korea dataset) ...")
    print(f"   target bbox = lon[{lon_min:.3f}, {lon_max:.3f}], "
          f"lat[{lat_min:.3f}, {lat_max:.3f}]")

    # Read EVERYTHING; we filter classes and clip spatially after reprojection.
    # Reading only the columns we need keeps memory reasonable.
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer,
                            columns=[code_field, "geometry"])
    except (TypeError, ValueError):
        # Older geopandas (<0.13) or fiona engines do not support the
        # `columns` argument; fall back to a full read.
        gdf = gpd.read_file(gpkg_path, layer=layer)

    print(f"   Total polygons in file : {len(gdf):,}")

    if code_field not in gdf.columns:
        raise ValueError(
            f"Field '{code_field}' not found in GPKG. "
            f"Available columns: {list(gdf.columns)}"
        )

    # Filter by class code FIRST (cheap)
    # The Korean MoE GPKG stores L2_CODE as strings ("110", "210", ...) but
    # the user passes them as integers. Match both forms to be robust across
    # different distributions of the dataset.
    classes = list(classes)
    classes_as_str = [str(c).strip() for c in classes]
    classes_as_int = []
    for c in classes:
        try:
            classes_as_int.append(int(c))
        except (ValueError, TypeError):
            pass

    # Diagnostic: print actual dtype and a few sample values
    sample = gdf[code_field].dropna().head(5).tolist()
    print(f"   {code_field} dtype = {gdf[code_field].dtype}, "
          f"sample values = {sample}")

    # Match against both integer and string forms
    mask_int = gdf[code_field].isin(classes_as_int)
    mask_str = gdf[code_field].astype(str).str.strip().isin(classes_as_str)
    gdf = gdf[mask_int | mask_str].copy()
    print(f"   After class filter     : {len(gdf):,}")

    if len(gdf) == 0:
        warnings.warn(
            "No polygons match the requested classes. Check that the GPKG's "
            f"{code_field} field contains the expected codes.")
        return np.zeros((H, W), dtype=bool)

    # Reproject to EPSG:4326 if needed
    if gdf.crs is None:
        warnings.warn(
            "GPKG has no CRS metadata; assuming EPSG:4326 (lat/lon).")
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    elif gdf.crs.to_epsg() != 4326:
        print(f"   Reprojecting from {gdf.crs} to EPSG:4326 ...")
        gdf = gdf.to_crs(epsg=4326)

    # Spatial clip in lat/lon to the satellite domain
    minx, miny, maxx, maxy = lon_min, lat_min, lon_max, lat_max
    bbox_polygon = gpd.GeoSeries.from_xy(
        [minx, maxx, maxx, minx], [miny, miny, maxy, maxy])
    # Use cx slicer (fast bbox spatial filter on the now-EPSG:4326 GeoDataFrame)
    gdf = gdf.cx[minx:maxx, miny:maxy]
    print(f"   After spatial clip     : {len(gdf):,}")

    if len(gdf) == 0:
        warnings.warn(
            "No polygons fall inside the satellite domain after reprojection. "
            "Check that lat/lon arrays are in degrees and that the GPKG "
            "actually covers your scene.")
        return np.zeros((H, W), dtype=bool)

    # Build the fine raster
    nx_fine = int(np.ceil((lon_max - lon_min) / fine_res_deg))
    ny_fine = int(np.ceil((lat_max - lat_min) / fine_res_deg))
    print(f"   Rasterizing to fine grid {ny_fine} x {nx_fine} "
          f"(~{fine_res_deg*111000:.0f} m) ...")

    transform = from_origin(lon_min, lat_max, fine_res_deg, fine_res_deg)

    fine_raster = features.rasterize(
        ((g, 1) for g in gdf.geometry if g is not None and not g.is_empty),
        out_shape=(ny_fine, nx_fine),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    n_fine_hit = int(fine_raster.sum())
    print(f"   Fine cells in rejection class: {n_fine_hit:,}")

    if n_fine_hit == 0:
        warnings.warn("Rasterization produced an empty raster.")
        return np.zeros((H, W), dtype=bool)

    # Aggregate to the GK2A grid
    print("   Aggregating fine raster to GK2A grid ...")

    # Build per-fine-cell centre coordinates
    fine_cols = np.arange(nx_fine)
    fine_rows = np.arange(ny_fine)
    fine_lon = lon_min + (fine_cols + 0.5) * fine_res_deg
    fine_lat = lat_max - (fine_rows + 0.5) * fine_res_deg

    # GK2A grid spacing (assumed approximately regular over Korea)
    if H > 1:
        dlat = (lat[-1, 0] - lat[0, 0]) / (H - 1)
    else:
        dlat = -fine_res_deg
    if W > 1:
        dlon = (lon[0, -1] - lon[0, 0]) / (W - 1)
    else:
        dlon = fine_res_deg

    if dlat == 0.0 or dlon == 0.0:
        raise ValueError(
            "GK2A grid spacing is zero - lat/lon arrays look constant.")

    gk_row_for_fine_lat = ((fine_lat - lat[0, 0]) / dlat).astype(np.int32)
    gk_col_for_fine_lon = ((fine_lon - lon[0, 0]) / dlon).astype(np.int32)

    valid_r = (gk_row_for_fine_lat >= 0) & (gk_row_for_fine_lat < H)
    valid_c = (gk_col_for_fine_lon >= 0) & (gk_col_for_fine_lon < W)

    GR, GC = np.meshgrid(gk_row_for_fine_lat, gk_col_for_fine_lon,
                         indexing="ij")
    VR, VC = np.meshgrid(valid_r, valid_c, indexing="ij")
    valid_grid = VR & VC

    counts_reject = np.zeros((H, W), dtype=np.int64)
    counts_total  = np.zeros((H, W), dtype=np.int64)

    flat_gr = GR[valid_grid].ravel()
    flat_gc = GC[valid_grid].ravel()
    flat_v  = fine_raster[valid_grid].ravel()

    np.add.at(counts_total,  (flat_gr, flat_gc), 1)
    np.add.at(counts_reject, (flat_gr, flat_gc), flat_v)

    coverage = np.where(counts_total > 0,
                        counts_reject / np.maximum(counts_total, 1),
                        0.0)

    mask = coverage >= coverage_threshold

    print(f"   GK2A pixels masked: {int(mask.sum())} / {H*W} "
          f"({100.0 * mask.sum() / (H*W):.1f} %)")
    return mask


# =============================================================================
# 4. Industrial heat-source mask  (ATBD Table 2.7)
# =============================================================================

# 18 industrial heat-source pixels of the Korean Peninsula listed in the
# GK2A FF ATBD v1.1, Table 2.7. Coordinates are approximate site centroids.
# The function below snaps each coordinate to the nearest grid pixel so the
# mask matches your specific lat/lon arrays.

_INDUSTRIAL_SITES_KR: List[Tuple[str, float, float]] = [
    ("POSCO Pohang steelworks 1",          36.002, 129.372),
    ("POSCO Pohang steelworks 2",          36.003, 129.381),
    ("POSCO Pohang steelworks 3",          36.010, 129.375),
    ("POSCO Pohang steelworks 4",          36.011, 129.382),
    ("POSCO Pohang steelworks 5",          36.015, 129.370),
    ("POSCO Gwangyang steelworks 1",       34.927, 127.703),
    ("POSCO Gwangyang steelworks 2",       34.935, 127.710),
    ("POSCO Gwangyang steelworks 3",       34.940, 127.700),
    ("POSCO Gwangyang steelworks 4",       34.945, 127.695),
    ("Hyundai Dangjin steelworks",         36.974, 126.648),
    ("Youngwol cement plant",              37.186, 128.468),
    ("Danyang cement plant",               37.003, 128.320),
    ("Donghae cement plant 1",             37.533, 129.124),
    ("Donghae cement plant 2",             37.527, 129.115),
    ("Yeosu industrial complex",           34.773, 127.744),
    ("Gori nuclear power plant",           35.313, 129.292),
    ("Gumi waste processing facility",     36.100, 128.330),
    ("Ulsan oil refinery",                 35.510, 129.363),
]


def build_industrial_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    sites: Optional[Iterable[Tuple[str, float, float]]] = None,
    max_match_km: float = 4.0,
    neighbourhood_radius_px: int = 2,
) -> np.ndarray:
    """
    Build a boolean mask of industrial heat-source pixels, with a small
    radius-neighbourhood expansion to catch nearby hot pixels that don't
    fall exactly on the listed site coordinates.

    Parameters
    ----------
    neighbourhood_radius_px : pixels around each site to also mask.
        2 -> 5x5 window, default. Use 0 to mask only exact pixels.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    H, W = lat.shape
    mask = np.zeros((H, W), dtype=bool)

    sites = list(sites) if sites is not None else _INDUSTRIAL_SITES_KR

    for name, slat, slon in sites:
        d2 = (lat - slat) ** 2 + (lon - slon) ** 2
        if not np.isfinite(d2).any():
            continue
        idx = np.unravel_index(np.nanargmin(d2), d2.shape)
        approx_km = np.sqrt(d2[idx]) * 111.0
        if approx_km > max_match_km:
            continue

        r0 = max(0, idx[0] - neighbourhood_radius_px)
        r1 = min(H, idx[0] + neighbourhood_radius_px + 1)
        c0 = max(0, idx[1] - neighbourhood_radius_px)
        c1 = min(W, idx[1] + neighbourhood_radius_px + 1)
        mask[r0:r1, c0:c1] = True

    return mask


# =============================================================================
# 5. Topographic reference pixels for lapse-rate estimation
# =============================================================================

def pick_topo_reference_pixels(
    dem: np.ndarray,
    land_mask: np.ndarray,
    cloud_mask: np.ndarray,
    n_pixels: int = 700,
    local_window: int = 21,
    min_local_std_m: float = 50.0,
    min_elevation_m: float = 100.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Choose ~700 reference pixels in regions of high DEM variability, to be
    used by the GK2A FF lapse-rate estimation step (ATBD Section 2.4.3).

    Strategy:
      1. Compute local DEM standard deviation in a `local_window`x`local_window`
         neighborhood of every pixel - this measures terrain roughness.
      2. Restrict to clear land pixels well above sea level
         (DEM > min_elevation_m).
      3. Pick the `n_pixels` pixels with highest local std.

    The ATBD uses a fixed table of 771 official reference points produced
    once per domain. Without that table this function gives a reasonable
    approximation: pixels in mountainous areas where ground temperature
    actually varies with elevation, suitable for OLS lapse-rate fitting.

    Parameters
    ----------
    dem            : (H, W) DEM [metres]
    land_mask      : (H, W) bool, True for land
    cloud_mask     : (H, W) bool, True for cloud
    n_pixels       : number of reference pixels to return
    local_window   : window size for local DEM std estimation (odd integer)
    min_local_std_m: minimum local std in metres to accept a pixel
    min_elevation_m: minimum DEM value to accept (avoids coastline)
    seed           : random seed - only used to break ties deterministically

    Returns
    -------
    ref_rows, ref_cols : 1-D int arrays of length up to n_pixels
                         (may be fewer if there are not enough qualifying pixels)
    """
    dem        = np.asarray(dem,        dtype=np.float64)
    land_mask  = np.asarray(land_mask,  dtype=bool)
    cloud_mask = np.asarray(cloud_mask, dtype=bool)

    if local_window % 2 == 0:
        local_window += 1

    # Mask out non-land / cloudy pixels with NaN before computing local std,
    # so the std reflects only land terrain
    dem_for_std = np.where(land_mask & ~cloud_mask, dem, np.nan)

    def _nanstd(v):
        v = v[np.isfinite(v)]
        if v.size < 2:
            return 0.0
        return float(np.std(v))

    local_std = generic_filter(dem_for_std, _nanstd, size=local_window,
                               mode="nearest")

    # Eligible pixels: clear land + above coastline + roughness > threshold
    eligible = (
        land_mask & (~cloud_mask) &
        np.isfinite(dem) &
        (dem >= min_elevation_m) &
        np.isfinite(local_std) &
        (local_std >= min_local_std_m)
    )

    # Add a tiny deterministic jitter to break ties so np.argsort is stable
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.0, 1e-6, size=local_std.shape)
    score = np.where(eligible, local_std + jitter, -np.inf)

    flat_idx = np.argsort(score, axis=None)[::-1]   # descending
    n_take = min(n_pixels, int(eligible.sum()))
    if n_take == 0:
        warnings.warn(
            "pick_topo_reference_pixels: no eligible pixels found. "
            "Lower min_local_std_m or check the land/cloud/DEM inputs."
        )
        return np.array([], dtype=int), np.array([], dtype=int)

    chosen = flat_idx[:n_take]
    ref_rows, ref_cols = np.unravel_index(chosen, score.shape)
    return ref_rows.astype(np.int32), ref_cols.astype(np.int32)


# =============================================================================
# 6. Convenience all-in-one builder
# =============================================================================

def build_all_inputs(
    lat: np.ndarray,
    lon: np.ndarray,
    dem: np.ndarray,
    land_mask: np.ndarray,
    cloud_mask: np.ndarray,
    scan_dt_utc: datetime,
    landcover_geotiff: Optional[str] = None,
    landcover_gpkg:    Optional[str] = None,
    landcover_array:   Optional[np.ndarray] = None,
    landcover_classes: Optional[Sequence[int]] = None,
    gpkg_code_field:   str = "L2_CODE",
    gpkg_layer:        Optional[str] = None,
    gpkg_coverage_threshold: float = 0.50,
    sub_lon: float = 128.2,
    n_topo_refs: int = 700,
    industrial_sites: Optional[Iterable[Tuple[str, float, float]]] = None,
) -> dict:
    """
    Convenience function that builds every missing input in one call.

    Returns a dict with keys:
        sza, vza, industrial_mask, baresoil_mask, topo_ref_rows, topo_ref_cols
    that can be unpacked directly into run_ff_algorithm().

    Landcover input is mutually exclusive (provide at most one):
      - landcover_gpkg    : path to a GPKG/SHP (e.g. Korean MoE landcover)
      - landcover_geotiff : path to a raster GeoTIFF (e.g. ESA WorldCover)
      - landcover_array   : pre-resampled array on the GK2A grid
      - none of the above : baresoil mask is all-False (no rejection)

    For the Korean MoE GPKG the default rejection classes are
    KOR_LC_REJECT_DEFAULT (urban + cropland + bare). For a GeoTIFF the
    default classes assume ESA WorldCover encoding (40, 50, 60).
    """
    out = {}

    print("[gk2a_inputs] Computing SZA from scan time + lat/lon ...")
    out["sza"] = compute_sza(lat, lon, scan_dt_utc)
    print(f"   SZA range : {np.nanmin(out['sza']):.1f} - "
          f"{np.nanmax(out['sza']):.1f} deg")

    print("[gk2a_inputs] Computing VZA from satellite geometry ...")
    out["vza"] = compute_vza(lat, lon, sub_lon=sub_lon)
    print(f"   VZA range : {np.nanmin(out['vza']):.1f} - "
          f"{np.nanmax(out['vza']):.1f} deg")

    print("[gk2a_inputs] Building industrial heat mask (ATBD Table 2.7) ...")
    out["industrial_mask"] = build_industrial_mask(
        lat, lon, sites=industrial_sites)
    print(f"   Industrial pixels masked: {int(out['industrial_mask'].sum())}")

    print("[gk2a_inputs] Building baresoil/urban mask ...")
    n_lc_sources = sum(x is not None for x in
                       (landcover_geotiff, landcover_gpkg, landcover_array))
    if n_lc_sources > 1:
        raise ValueError(
            "Provide AT MOST one of landcover_geotiff / landcover_gpkg / "
            "landcover_array."
        )

    if landcover_gpkg is not None:
        classes = (landcover_classes if landcover_classes is not None
                   else KOR_LC_REJECT_DEFAULT)
        out["baresoil_mask"] = build_baresoil_mask_from_gpkg(
            landcover_gpkg, lat, lon,
            classes=classes,
            code_field=gpkg_code_field,
            layer=gpkg_layer,
            coverage_threshold=gpkg_coverage_threshold,
        )
        print(f"   Loaded from GPKG: {landcover_gpkg}")
    elif landcover_geotiff is not None:
        classes = (landcover_classes if landcover_classes is not None
                   else (40, 50, 60))
        out["baresoil_mask"] = build_baresoil_mask_from_geotiff(
            landcover_geotiff, lat, lon, classes=classes)
        print(f"   Loaded from GeoTIFF: {landcover_geotiff}")
    elif landcover_array is not None:
        classes = (landcover_classes if landcover_classes is not None
                   else (40, 50, 60))
        out["baresoil_mask"] = build_baresoil_mask_from_array(
            landcover_array, classes=classes)
        print(f"   Built from provided array.")
    else:
        out["baresoil_mask"] = np.zeros(lat.shape, dtype=bool)
        warnings.warn(
            "build_all_inputs: no landcover input provided; baresoil mask "
            "is all-False. False alarms over cropland and urban areas may "
            "persist."
        )
    print(f"   Baresoil pixels masked  : {int(out['baresoil_mask'].sum())}")

    print("[gk2a_inputs] Selecting topographic reference pixels ...")
    rr, cc = pick_topo_reference_pixels(
        dem, land_mask, cloud_mask, n_pixels=n_topo_refs)
    out["topo_ref_rows"] = rr
    out["topo_ref_cols"] = cc
    print(f"   Selected {len(rr)} reference pixels.")

    return out


# =============================================================================
# Self-test when run as a script
# =============================================================================

if __name__ == "__main__":
    print("Running self-test ...")

    H, W = 60, 60
    lat = np.linspace(38.0, 35.0, H)[:, None] * np.ones((1, W))
    lon = np.linspace(126.5, 129.5, W)[None, :] * np.ones((H, 1))

    sza = compute_sza(lat, lon, datetime(2024, 4, 4, 5, 0))   # 14:00 KST
    print(f"  SZA   : {sza.min():.1f} - {sza.max():.1f} deg "
          "(expect daytime values)")
    assert (sza < 80).all(), "Expected daytime SZA values"

    vza = compute_vza(lat, lon, sub_lon=128.2)
    print(f"  VZA   : {vza.min():.2f} - {vza.max():.2f} deg "
          "(expect ~35-45 over Korea)")
    assert (vza < 60).all() and (vza > 30).all()

    ind_mask = build_industrial_mask(lat, lon)
    print(f"  Industrial pixels found in grid: {int(ind_mask.sum())} "
          f"(out of {len(_INDUSTRIAL_SITES_KR)} sites)")
    assert ind_mask.sum() > 0

    rng = np.random.default_rng(0)
    fake_dem = (rng.normal(0, 200, (H, W)) +
                500 * np.exp(-((np.arange(H)[:,None]-30)**2 +
                               (np.arange(W)[None,:]-30)**2)/200))
    rr, cc = pick_topo_reference_pixels(
        fake_dem, np.ones((H, W), dtype=bool), np.zeros((H, W), dtype=bool),
        n_pixels=50, local_window=11, min_local_std_m=30, min_elevation_m=0)
    print(f"  Topo refs picked : {len(rr)}")
    assert len(rr) == 50

    print("\nAll checks passed.")