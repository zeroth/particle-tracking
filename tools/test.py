import dask.array as da
import os
import tifffile

# path = "D:/Data/Sudipta/Arpan/ML/Data/exp1"
# arr = da.from_zarr(os.path.join(path, "ord_data.zarr"))
# os.makedirs(os.path.join(path, "org"), exist_ok=True)
# for i in range(arr.shape[0]):
#     tifffile.imwrite(os.path.join(path, "org",f"org_{i}.tiff"), arr[i])

import numpy as np
from multiprocessing import Pool

def add(x):
    return x + 1

if __name__ == '__main__':
    a = np.zeros((3,2,2), dtype='uint8')

    for i in range(a.shape[0]):
        a[i] = i+1

    pool = Pool(processes=4)
    x = pool.map(add, [a[i] for i in range(a.shape[0])])
    x = np.array(x)
    print(a.shape)
    print(x.shape)

