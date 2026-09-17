import numpy as np

bin_path = '/home/xurshedjonff/gk2a_ff/land_mask/lsmask_2km.bin'
out_path = '/home/xurshedjonff/gk2a_ff/land_mask/land_mask_fd_2km.npy'

arr = np.fromfile(bin_path, dtype=np.int16).reshape(5500, 5500)

print('shape:', arr.shape)
print('dtype:', arr.dtype)
print('unique values:', np.unique(arr))

# 1 = land
# 0 = sea
# -999 = fill / invalid
land_mask = (arr == 1)

np.save(out_path, land_mask.astype(bool))
print('saved:', out_path)