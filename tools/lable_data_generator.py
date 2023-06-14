from skimage import data
from skimage.exposure import histogram
import tifffile
import matplotlib.pyplot as plt
import os
import sys
from skimage import data, feature, exposure
import numpy as np
from skimage.filters import threshold_otsu
import threading
import logging
import time
from pathlib import Path
from tqdm import tqdm

# def scale_to(x, x_min, x_max, dtype):
#     r = x_max - x_min
#     t_max = np.iinfo(np.dtype(dtype)).max
#     # assert(math.isclose(0,r, abs_tol=np.finfo(float).eps) == False)
#     x_s =  ((x - x_min) / r) * t_max
#     return x_s
# path = "D:\\Data\\Sudipta\\Arpan"
# file_name = "single_frame_send_1.tif"


def scale_to(x,  dtype):
    r = np.max(x) - np.min(x)
    try:
        t_max = np.iinfo(np.dtype(dtype)).max
    except:
        t_max = np.finfo(np.dtype(dtype)).max
    
    # assert(math.isclose(0,r, abs_tol=np.finfo(float).eps) == False)
    x_s =  ((x - np.min(x)) / r) * t_max
    return x_s.astype(dtype)

def scale_to_float(x,  dtype_in):
#     r = np.max(x) - np.min(x)
    try:
        i_min = np.iinfo(np.dtype(dtype_in)).min
        i_max = np.iinfo(np.dtype(dtype_in)).max
    except:
        i_min = np.finfo(np.dtype(dtype_in)).min
        i_max = np.finfo(np.dtype(dtype_in)).max
    
    r = i_max -i_min
    return (x - i_min)/r

def subtract(a, b, dtype='uint16'):
    try:
        t_min = np.iinfo(np.dtype(dtype)).min
        t_max = np.iinfo(np.dtype(dtype)).max
    except:
        t_min = np.finfo(np.dtype(dtype)).min
        t_max = np.finfo(np.dtype(dtype)).max        
    return np.clip(a.astype('int32') - b, t_min, t_max).astype(a.dtype)

def top_hat(image, radius=50, light_bg=False):
        from skimage.morphology import white_tophat, black_tophat, disk
        str_el = disk(radius) #you can also use 'ball' here to get a slightly smoother result at the cost of increased computing time
        if light_bg:
            return  scale_to(black_tophat(image, str_el), image.dtype)
        else:
            return  scale_to(white_tophat(image, str_el), image.dtype)

def find_points(image):
    # equalize_hist_img = exposure.equalize_hist(image)
    thresh = threshold_otsu(image)
    thresh_img = subtract(image, thresh, image.dtype)
    top_hat_img = top_hat(thresh_img)
    blobs = feature.blob_log(top_hat_img, num_sigma= 100, overlap=0.9, threshold=0.1)
    # return blobs[:, :2]
    return blobs

from skimage.draw import disk
from math import sqrt
def draw_points(image, points, radius=1):
    # points = points[:,:2]
    def map_bound(limit):
        def fun(val):
            # logging.info("befor: limit %d. val %d", limit, val)
            if val >= limit: 
                val = limit-1
            elif val < 0:
                val = 0
            # logging.info("after: limit %d. val %d", limit, val)
            return val
        return fun

    for x, y, r in points:
        rr, cc = disk((x,y), radius=r*sqrt(2))
        rr = np.array(list(map(map_bound(image.shape[0]), rr)), dtype='uint16')
        cc = np.array(list(map(map_bound(image.shape[1]), cc)), dtype='uint16')
        image[rr, cc] = 255
    return image

def get_mask(input, result, index):
    logging.info("Processing %d ", index)
    points = find_points(input)
    mask = draw_points(result[index], points)
    result[index] = mask

def get_mask_v2(input, radius=1):
    points = find_points(input)
    return draw_points(np.zeros(input.shape, dtype=input.dtype), points, radius)

from multiprocessing import Pool
def main(input, radius=1, prefix=""):
    # data  = tifffile.imread(os.path.join(path, "*.tif*"))
    data = tifffile.imread(path)
    result = np.zeros(data.shape, dtype=data.dtype)
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
    
    logging.info("Starting the threading")

    processes = int(os.cpu_count()-1) if os.cpu_count() >=8 else os.cpu_count()/2
    
    pool = Pool(processes= processes)
    x = pool.starmap(get_mask_v2, [(data[i], radius) for i in range(data.shape[0])])
    result = np.array(x)

    logging.info("Saving mask")
    input = Path(input)
    output_path = os.path.join(input.parent,  "mask_{0}_{1}".format(prefix, input.name))
    print(output_path)
    print(result.shape)
    tifffile.imwrite(output_path, result)

    ##### Thread version ###
    # threads = list()
    # for i in range(data.shape[0]):
    # # for i in range(1000,1010):
    #     t = threading.Thread(target=get_mask, args=(data[i], result, i))
    #     threads.append(t)
    #     t.start()

    # logging.info("Joining the threading")
    # for t in threads:
    #     t.join()
    #     # get_mask(data[i], result, i)

# def test(input, prefix=""):
#     # data  = tifffile.imread(os.path.join(path, "*.tif*"))
#     data = tifffile.imread(path)
#     result = np.zeros(data.shape, dtype=data.dtype)
#     logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
    
#     logging.info("Starting the test")
#     # mask = get_mask_v2, [data[i] for i in range(data.shape[0])]
#     points = find_points(data)
#     print(points)
#     logging.info("End")
#     mask = get_mask_v2(data)
    
#     logging.info("Saving mask")
#     input = Path(input)
#     output_path = os.path.join(input.parent, "mask_{0}_{1}".format(prefix, input.name))
#     print(output_path)
#     print(result.shape)
#     tifffile.imwrite(output_path, mask)

# def init_worker(result, path):
#     global result_image
#     # global result_path
#     result_image = result
#     # result_path = path

# def get_mask_v3(input, radius=1, index=0):
#     points = find_points(input)
#     mask =  draw_points(np.zeros(input.shape, dtype=input.dtype), points, radius)
#     global result_image
#     # global result_path
#     result_image[index] = mask
#     result_image.flush()
#     # tifffile.imwrite(result_path, result_image)
#     print("saved {0}".format(index))
    
# def main_v2(input, radius=1, prefix=""):
#     logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")

#     data = tifffile.imread(path)
    
#     result = np.zeros(data.shape, dtype=data.dtype)
    
#     input = Path(input)
#     output_path = os.path.join(input.parent,  "mask_{0}_{1}".format(prefix, input.name))
#     tifffile.imwrite(output_path, result)
#     del result
#     result = tifffile.memmap(output_path, shape=data.shape, dtype=data.dtype, mode='r+')
    
#     logging.info("Starting the threading")

#     processes = int(os.cpu_count()-1) if os.cpu_count() >=8 else os.cpu_count()/2
    
#     with Pool(initializer=init_worker, initargs=(result, output_path, ), processes=processes) as pool:
#         pool.starmap(get_mask_v3, [(data[i], radius, i) for i in range(data.shape[0])])
    
#     result.flush()
#     # pool = Pool(processes= processes)
#     # x = pool.starmap(get_mask_v2, [(data[i], radius) for i in range(data.shape[0])])
#     # result = np.array(x)

#     logging.info("Done !")
#     print(output_path)
#     print(result.shape)
#     del result
    


import argparse
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]..."
    )

    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('dir', nargs=1)
    parser.add_argument('-r', '--radius', nargs=1, type=float, default=1)
    time_str = time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime())
    parser.add_argument('-p', '--prefix', nargs=1, type=str, default=time_str)
    return parser

if __name__ == "__main__":
    
    parser = init_argparse()
    # args = parser.parse_args()
    args = parser.parse_args("E:/Sudipta/Arpan/send-1.tif -p radius_sqrt ".split())
    # args = parser.parse_args("D:/Data/Sucharita/C1-20ul_liposome_with_high_peptide_30ms_33.33hz_1_ch2_left.tif -p radius_sqrt ".split())
    # args = parser.parse_args("D:/Data/Sucharita/sample/C1-20ul_liposome_with_high_peptide_30ms_33.33hz_1_ch2_left_10.tif -p radius_sqrt_sample ".split())
    print(args)

    path = args.dir[0]
    radius = args.radius
    prefix = args.prefix[0]
    # # path = sys.argv[1]
    # # radius = 1
    # # if len(sys.argv) > 2:
        
    main(path, radius=radius, prefix=prefix)
    # test(path, prefix=prefix)