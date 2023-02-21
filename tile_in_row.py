from aicsimageio.readers import CziReader
import numpy as np
smear = "extern_Synlab_2151_26_3_MTB.czi"
reader = CziReader(f"/mnt/storage/TBProject/TB_sample/2022-06-29/{smear}")

meta = np.zeros((1563, 3))
current_line = 0
meta[0, 2] = 0
for i in range(0, 1563):
    x, y = reader.get_mosaic_tile_position(i)
    meta[i, 0] = x
    meta[i, 1] = y
    meta[i, 2] = current_line
    if i > 0:
        if np.abs(meta[i,0] - meta[i-1,0]) > 1000:
            current_line += 1
            meta[i, 2] = current_line



# save meta data as npy file
np.save(f"TB_sample/meta_{smear}.npy", meta)
