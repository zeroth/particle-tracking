import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import tifffile
import time

def main():
    path = "Z:/BDI/Laura/20230615_163804"
    data = tifffile.imread(os.path.join(path, "*.tif"))
    print(data.shape)
    
    
    start = 2310
    end = 2320
    step = end - start
    s = 0
    e = step
    result = np.zeros((data.shape[-2], int(step*data.shape[0])+5))

    for i in range(data.shape[0]):
        print(s, e)
        result[:, s:e] = data[i][:, start:end]
        s = e
        e = s+step
    
    time_str = time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime())
    tifffile.imwrite(os.path.join(path, f"result_{time_str}.tif"), result)


if __name__ == "__main__":
    main()