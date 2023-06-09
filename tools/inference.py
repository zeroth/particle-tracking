import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2" #https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

import sys
from pathlib import Path
from typing import Optional
import argparse
import time

import random
import numpy as np
import dask.array as da

# https://pytorch.org/docs/stable/notes/randomness.html
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

from torch import nn, optim

from particle_tracking.data import Data2D, Dataset2D
from particle_tracking.model import Model
from particle_tracking.utils import reshape_input_data, revert_tensor_shape, reshape_output_data, convert_tensor_shape, load_tiff, iou, accuracy, float_to_unit8

import tifffile
from tqdm import tqdm
from pprint import pprint
import logging
from datetime import datetime

def main(images_path, output_path, model_path):
    
    images = load_tiff(images_path)
    os.makedirs(output_path, exist_ok=True)
    time_str = time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime())
    output_file_path = os.path.join(output_path, f"mask_{time_str}.tif")
    pred = da.zeros_like(images, chunks=images.chunksize, dtype=np.uint16)

    model = Model(
                pre_trained_model_path=model_path,
                optimizerCls=optim.Adam,
                loss_fn=nn.BCEWithLogitsLoss(),
                encoder_name="resnet18", 
                encoder_weights="imagenet",
                in_channels = 1, 
                classes=1,
                is_inference_mode=True )
    
    logging.info(f"starting loop {images.shape}")
    for i in tqdm(range(images.shape[0]), desc="Inference Loop"):
        img = images[i].compute()
        mask = model.inference(img=img)
        pred[i] = mask
    
    logging.info(f"saving to {output_file_path} - {pred.shape}")
    tifffile.imwrite(output_file_path, pred)



def init_argparse() -> argparse.ArgumentParser:
    logging.basicConfig(format="%(asctime)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]..."
    )

    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('images', nargs='?', const=str)
    parser.add_argument('labels', nargs='?', const=str)
    parser.add_argument('-im','--inputmodel', nargs='?', const=str, default=None)
    
    # time_str = time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime())
    # parser.add_argument('-p', '--prefix', nargs=1, type=str, default=time_str)
    return parser


if __name__ == "__main__":
     # https://pytorch.org/docs/stable/notes/randomness.html
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    da.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    parser = init_argparse()
    args = parser.parse_args("D:/Data/Sudipta/Arpan/send-1.tif D:/Data/Sudipta/Arpan/op -im D:/Lab/particle-tracking/model_radius_sqrt_new_train/model_final_14_06_2023_11_31_30.pt".split())
    _MODEL_FILE_NAME_ = 'model_final.pt'
    _MODEL_DIR_ = Path.home().joinpath('.ml_particle_tracking')
    _MODEL_FILE_PATH_ = _MODEL_DIR_.joinpath(_MODEL_FILE_NAME_)
    print(f" exists : {os.path.exists(_MODEL_FILE_PATH_)}, {_MODEL_FILE_PATH_}")
    # args = parser.parse_args("D:/Data/Sudipta/Arpan/ML_Training/data/send-1.tif D:/Data/Sudipta/Arpan/ML_Training/label/mask_radius_sqrt.tif -om ./model_radius_sqrt_new_train".split())
    logging.info(f"Main Args : {args}")
    print("test")
    labels = args.labels
    images = args.images
    input_model = _MODEL_FILE_PATH_
    main(images_path=images, output_path=labels, model_path=input_model)
