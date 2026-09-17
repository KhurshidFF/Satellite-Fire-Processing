import netCDF4 as nc
import numpy as np
from pyproj import CRS, Transformer

PRE = '/home/xurshedjonff/gk2a_ff/data/preprocessed/hamyang_case/gk2a_ff_4ch_202602211930_preprocessed_FD.nc'

# --- official CGMS constants (from the L1B file) ---
sub_lon_deg = np.rad2deg(2.2375121010567303)   # 128.2002 deg
H_surface   = 42164000.0 - 6378137.0
CFAC = 20425338.903339352
LFAC = -20425338.903339352
COFF = LOFF = 2750.5
N = 5500

crs_geos = CRS.from_proj4(
    f"+proj=geos +lon_0={sub_lon_deg} +h={H_surface} "
    f"+a=6378137.0 +b=6356752.3 +sweep=y +units=m +no_defs")
tf = Transformer.from_crs(crs_geos, CRS.from_epsg(4326), always_xy=True)

# Full-grid lat/lon using the SAME convention that matched pyproj above
cols = np.arange(N) + 1.0          # CGMS 1-based
rows = np.arange(N) + 1.0
x_rad = np.deg2rad((cols - COFF) / (2.0**-16 * CFAC))
y_rad = np.deg2rad((rows - LOFF) / (2.0**-16 * LFAC))
X = x_rad * H_surface
Y = y_rad * H_surface
XX, YY = np.meshgrid(X, Y)
LON_new, LAT_new = tf.transform(XX, YY)   # note: transform returns (lon, lat) for always_xy
# fix off-disk points
off = ~np.isfinite(LON_new) | ~np.isfinite(LAT_new) | (np.abs(LON_new) > 180) | (np.abs(LAT_new) > 90)
LON_new[off] = np.nan
LAT_new[off] = np.nan

# --- sanity check before writing ---
ds = nc.Dataset(PRE, 'r')
LAT_old = np.ma.filled(ds.variables['latitude'][:], np.nan)
LON_old = np.ma.filled(ds.variables['longitude'][:], np.nan)
ds.close()
for (r,c) in [(2750,2750),(970,2710),(900,2650)]:
    print(f"({r},{c}) old {LAT_old[r,c]:.4f},{LON_old[r,c]:.4f}  "
          f"new {LAT_new[r,c]:.4f},{LON_new[r,c]:.4f}")

# --- write corrected lat/lon back into the preprocessed file ---
# (make a backup copy of the file first!)
ds = nc.Dataset(PRE, 'a')                 # append/modify mode
ds.variables['latitude'][:]  = LAT_new.astype(ds.variables['latitude'].dtype)
ds.variables['longitude'][:] = LON_new.astype(ds.variables['longitude'].dtype)
ds.geolocation_note = ("latitude/longitude regenerated with CGMS GEOS "
                       "convention (1-based, COFF/LOFF=2750.5) to match "
                       "official GK2A grid")
ds.close()
print("Corrected lat/lon written.")