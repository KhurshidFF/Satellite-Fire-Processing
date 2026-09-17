import netCDF4 as nc
import numpy as np

# -------------------------
# Input / output paths
# -------------------------
in_path = '/home/xurshedjonff/gk2a_ff/cloud_mask/gk2a_ami_le2_cld_fd020ge_202602222140.nc'
out_path = '/home/xurshedjonff/gk2a_ff/cloud_mask/cloud_mask_fd_202602222140.npy'

# -------------------------
# Read CLD file
# -------------------------
ds = nc.Dataset(in_path)
cld = ds.variables['CLD'][:]
ds.close()

print('shape:', cld.shape)
print('dtype:', cld.dtype)
print('unique values:', np.unique(cld))

# Convert masked array -> normal ndarray
# Fill masked pixels as 1 (= probably cloud), which is safer than clear
if np.ma.isMaskedArray(cld):
    cld = cld.filled(1)

# -------------------------
# Convert to boolean cloud mask
# 0 = cloud
# 1 = probably cloud
# 2 = clear
# True = cloud / not-clear
# False = clear
# -------------------------
cloud_mask = (cld != 2)

# -------------------------
# Save
# -------------------------
np.save(out_path, np.asarray(cloud_mask, dtype=bool))

print('saved:', out_path)
print('cloud mask unique:', np.unique(cloud_mask))