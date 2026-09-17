#!/usr/bin/env python
# -*- coding: utf-8 -*-

import netCDF4 as nc
import numpy as np
import rasterio


# -------------------------------------------------
# 1. Paths
# -------------------------------------------------
target_nc_path = '/home/xurshedjonff/gk2a_ff/data/preprocessed/hamyang_case/gk2a_ff_4ch_202602211930_preprocessed_FD.nc'
dem_tif_path = '/home/xurshedjonff/gk2a_ff/dem/south_korea_srtm_30_meter.tif'

output_npy_path = '/home/xurshedjonff/gk2a_ff/dem/dem_fd_2km_202602211930.npy'
output_nc_path = '/home/xurshedjonff/gk2a_ff/dem/dem_fd_2km_202602211930.nc'


# -------------------------------------------------
# 2. Read target GK2A lat/lon grid
# -------------------------------------------------
ds = nc.Dataset(target_nc_path, 'r')
lat = ds.variables['latitude'][:]
lon = ds.variables['longitude'][:]
ds.close()

print('Target grid shape:', lat.shape)

# Flatten for sampling
lat_flat = lat.ravel()
lon_flat = lon.ravel()

# Valid geographic points only
valid_geo = np.isfinite(lat_flat) & np.isfinite(lon_flat)

# Prepare output
dem_flat = np.full(lat_flat.shape, np.nan, dtype=np.float32)


# -------------------------------------------------
# 3. Read DEM and sample to GK2A grid
# -------------------------------------------------
with rasterio.open(dem_tif_path) as src:
    nodata = src.nodata
    print('DEM CRS:', src.crs)
    print('DEM size:', src.width, src.height)
    print('DEM nodata:', nodata)

    # rasterio.sample expects (x, y) = (lon, lat)
    coords = list(zip(lon_flat[valid_geo], lat_flat[valid_geo]))

    sampled = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)

    if nodata is not None:
        sampled[sampled == nodata] = np.nan

    dem_flat[valid_geo] = sampled


# -------------------------------------------------
# 4. Reshape back to GK2A grid
# -------------------------------------------------
dem_grid = dem_flat.reshape(lat.shape)

# For FD testing with Korea-only DEM:
# fill missing pixels outside DEM coverage with 0.0
# (acceptable for Hamyang-focused testing, not strict full-disk replication)
dem_grid_filled = np.where(np.isfinite(dem_grid), dem_grid, 0.0).astype(np.float32)

print('DEM min/max on grid:', np.nanmin(dem_grid), np.nanmax(dem_grid))
print('DEM filled min/max:', dem_grid_filled.min(), dem_grid_filled.max())


# -------------------------------------------------
# 5. Save as NPY
# -------------------------------------------------
np.save(output_npy_path, dem_grid_filled)
print('Saved:', output_npy_path)


# -------------------------------------------------
# 6. Save as NetCDF
# -------------------------------------------------
out = nc.Dataset(output_nc_path, 'w', format='NETCDF4')
out.createDimension('y', dem_grid_filled.shape[0])
out.createDimension('x', dem_grid_filled.shape[1])

v_dem = out.createVariable('dem_m', np.float32, ('y', 'x'))
v_lat = out.createVariable('latitude', np.float32, ('y', 'x'))
v_lon = out.createVariable('longitude', np.float32, ('y', 'x'))

v_dem[:] = dem_grid_filled
v_lat[:] = lat
v_lon[:] = lon

out.description = 'DEM resampled to GK2A target grid'
out.source_dem = dem_tif_path
out.target_grid = target_nc_path
out.close()

print('Saved:', output_nc_path)