import json
import netCDF4 as nc
import numpy as np
from datetime import datetime
from gk2a_ff_algorithm import run_ff_algorithm, summarise, make_fire_report
from gk2a_inputs import build_all_inputs, KOR_LC_REJECT_DEFAULT

# -------------------------
# Paths
# -------------------------
pre_nc    = '/home/xurshedjonff/gk2a_ff/data/preprocessed/hamyang_case/gk2a_ff_4ch_202602211930_preprocessed_FD.nc'
land_npy  = '/home/xurshedjonff/gk2a_ff/land_mask/land_mask_fd_2km.npy'
cloud_npy = '/home/xurshedjonff/gk2a_ff/cloud_mask/cloud_mask_fd_202602211930.npy'
dem_npy   = '/home/xurshedjonff/gk2a_ff/dem/dem_fd_2km_202602211930.npy'
gpkg_path = '/home/xurshedjonff/gk2a_ff/land_cover/korea_landcover_2025_medium.gpkg'
out_dir   = '/home/xurshedjonff/gk2a_ff/outputs'

# -------------------------
# Read full-disk arrays
# -------------------------
ds = nc.Dataset(pre_nc)
T07_full = ds.variables['bt_sw038'][:]
T14_full = ds.variables['bt_ir112'][:]
R03_full = ds.variables['refl_vi006'][:]
R04_full = ds.variables['refl_vi008'][:]
LAT_full = ds.variables['latitude'][:]
LON_full = ds.variables['longitude'][:]
ds.close()

LAND_full  = np.load(land_npy).astype(bool)
CLOUD_full = np.load(cloud_npy).astype(bool)
DEM_full   = np.load(dem_npy).astype(np.float32)

# -------------------------
# CROP TO KOREAN PENINSULA
# -------------------------
LAT_MIN, LAT_MAX = 33.0, 39.5
LON_MIN, LON_MAX = 124.0, 132.0

LAT_arr = np.ma.filled(LAT_full, np.nan).astype(np.float64)
LON_arr = np.ma.filled(LON_full, np.nan).astype(np.float64)

in_box = ((LAT_arr >= LAT_MIN) & (LAT_arr <= LAT_MAX) &
          (LON_arr >= LON_MIN) & (LON_arr <= LON_MAX))
if not in_box.any():
    raise RuntimeError("No pixels found inside the Korean bounding box. "
                       "Check that LAT/LON arrays are in degrees.")

rows_any = in_box.any(axis=1)
cols_any = in_box.any(axis=0)
r0 = int(np.argmax(rows_any))
r1 = int(len(rows_any) - np.argmax(rows_any[::-1]))
c0 = int(np.argmax(cols_any))
c1 = int(len(cols_any) - np.argmax(cols_any[::-1]))

print(f"Cropping FD {LAT_full.shape} -> Korea slice "
      f"rows[{r0}:{r1}] cols[{c0}:{c1}] -> shape ({r1-r0}, {c1-c0})")

sl = (slice(r0, r1), slice(c0, c1))
T07   = T07_full[sl]
T14   = T14_full[sl]
R03   = R03_full[sl]
R04   = R04_full[sl]
LAT   = np.asarray(LAT_arr[sl], dtype=np.float64)
LON   = np.asarray(LON_arr[sl], dtype=np.float64)
LAND  = LAND_full[sl]
CLOUD = CLOUD_full[sl]
DEM   = DEM_full[sl]

print(f"  LAT range : {np.nanmin(LAT):.2f} - {np.nanmax(LAT):.2f}")
print(f"  LON range : {np.nanmin(LON):.2f} - {np.nanmax(LON):.2f}")

# -------------------------
# Build all auxiliary inputs
# -------------------------
# Filename 202602222140 = 2026-02-22 21:40 UTC (nighttime over Korea, SZA ~95 deg)
scan_dt = datetime(2026, 2, 22, 21, 40)   # UTC

aux = build_all_inputs(
    lat=LAT, lon=LON, dem=DEM,
    land_mask=LAND, cloud_mask=CLOUD,
    scan_dt_utc=scan_dt,
    landcover_gpkg=gpkg_path,
    landcover_classes=KOR_LC_REJECT_DEFAULT,
    gpkg_code_field="L2_CODE",
    gpkg_coverage_threshold=0.5,
)

PREV_DQF = None

# -------------------------
# Run the algorithm
# -------------------------
result = run_ff_algorithm(
    T_SWIR=T07, T_TIR=T14,
    refl_red=R03, refl_nir=R04,
    sza=aux["sza"],
    vza=aux["vza"],
    land_mask=LAND, cloud_mask=CLOUD, dem=DEM,
    industrial_mask=aux["industrial_mask"],
    baresoil_mask=aux["baresoil_mask"],
    topo_ref_rows=None,   # fallback -7/-6 K/km (auto-estimate gave unphysical -10 K/km)
    topo_ref_cols=None,
    prev_dqf=PREV_DQF,
    use_theoretical_urban_rejection=False,
)

print(summarise(result))
fire_pixels = make_fire_report(result.FF, LAT, LON)
print('number of fire pixels =', len(fire_pixels))
print('first 20 fire pixels =', fire_pixels[:20])

# -------------------------
# NO empirical shift any more.
# The geolocation offset was fixed at its source by regenerating the
# latitude/longitude grid with the correct CGMS GEOS convention. The
# coordinates returned by make_fire_report are now the corrected ones.
# -------------------------

# -------------------------
# Export fire points to GeoJSON (corrected grid, no shift)
# -------------------------
features = []
for (r, c, plat, plon) in fire_pixels:
    if not (np.isfinite(plat) and np.isfinite(plon)):
        continue
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [plon, plat]},
        "properties": {"row": int(r), "col": int(c), "fire": 1},
    })
geojson = {"type": "FeatureCollection",
           "name": "gk2a_ff_202602222140",
           "features": features}
with open(f'{out_dir}/FF_202602222140.geojson', 'w', encoding='utf-8') as f:
    json.dump(geojson, f, ensure_ascii=False)
print(f'saved {len(features)} fire points -> {out_dir}/FF_202602222140.geojson')

# -------------------------
# Save outputs (cropped grid) + crop offsets + corrected LAT/LON
# -------------------------
np.save(f'{out_dir}/FF_binary_202602211930.npy',        result.FF)
np.save(f'{out_dir}/DQF_FF_202602211930.npy',           result.DQF_FF)
np.save(f'{out_dir}/T_SWIR_corrected_202602211930.npy', result.T_SWIR_c)
np.save(f'{out_dir}/T_TIR_corrected_202602211930.npy',  result.T_TIR_c)
np.save(f'{out_dir}/delta_T_202602211930.npy',          result.delta_T)
np.save(f'{out_dir}/base_SWIR_202602211930.npy',        result.base_SWIR)
np.save(f'{out_dir}/base_dT_202602211930.npy',          result.base_dT)
np.save(f'{out_dir}/crop_offsets_202602211930.npy',     np.array([r0, r1, c0, c1]))
# Save the corrected crop LAT/LON so downstream scripts never re-read the grid
np.save(f'{out_dir}/LAT_202602211930.npy', LAT)
np.save(f'{out_dir}/LON_202602211930.npy', LON)
print("saved all .npy outputs")