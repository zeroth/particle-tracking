from typing import Optional

import numpy as np
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm
import tifffile
import os
import dask.array as da


class SegmentationModel:
    def __init__(
        self, checkpoint: str = None) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.model = load_model(self.device, checkpoint)
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels = 1,
            classes=1
        )
        self.model.eval()
        self.model.to(self.device)

        if checkpoint:
            print("loading checkpoint")
            checkpoint_state = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint_state["state_dict"])
        
            # for param_tensor in checkpoint_state.items():
            #     print(param_tensor[0])
            # print(checkpoint_state["state_dict"])

    def pre_process(self, img: np.ndarray) -> torch.Tensor:
        """Pre-process the image for inference"""

        # convert to tensor
        img_t = torch.FloatTensor(img)
        
        # add batch dimensions
        if img_t.ndim == 3:
            # batch 2D image, add channel 
            img_t = img_t.reshape(shape=(img_t.shape[0], 1, img_t.shape[1], img_t.shape[2])) 
        else:
            # 2D image only, add batch and channel
            img_t = img_t.reshape(shape=(1, 1, img_t.shape[0], img_t.shape[1]))

        # move to device
        img_t = img_t.to(self.device)

        return img_t

    def inference(self, img: np.ndarray) -> np.ndarray:
        """Run model inference on the input image"""
        with torch.no_grad():
            img_t = self.pre_process(img)

            # inference
            output = self.model(img_t)

            # post-process
            mask = self.post_process_image(output)
            
            return mask
        
    def post_process_image(self, output: torch.Tensor) -> np.ndarray:
        """Post process model output into binary mask."""
        out1 = torch.sigmoid(output)
        out1 = (out1 > 0.5).float()
        pred = out1 * 255.0

        pred = pred.detach().cpu().numpy()

        return pred
    

def tiff_to_array(path):
    arr = tifffile.imread(os.path.join(path, "*.tif*"))
    return arr.astype(np.float32)

def load_2d_data(path, shape_limit = (640, 640)):
    arr = tiff_to_array(path)

    print("input data shape", arr.shape)
    if arr.ndim < 2:
        raise Exception("Data has to be minumum two dimintional you have provided only one")
    
     # add z dim for 2d arr
    # if arr.ndim == 2:
    #     arr = da.reshape(arr, shape=(1, arr.shape[0], arr.shape[1]))

    # check last two dims of data and get them to the shape limit
    if(arr.shape[-2] > shape_limit[0] or arr.shape[-1] > shape_limit[1]):
        max_shape = np.max([arr.shape[-2], arr.shape[-1]])
        limit = int(np.ceil(max_shape/32)*32 ) # upper multiple of 32
        shape_limit = (limit, limit)
    
    pad_0 = (shape_limit[0] - arr.shape[-2]) // 2
    pad_1 = (shape_limit[1] - arr.shape[-1]) // 2

    arr = da.pad(arr, ((0,0), (pad_0, pad_0), (pad_1,pad_1)), mode="constant", constant_values = 0)

    pad_1, pad_2 = 0, 0
    if arr.shape[-2] % 2 != 0:
        pad_1 = 1
    if arr.shape[-1] % 2 != 0:
        pad_2 = 1
    
    arr = da.pad(arr, ((0, 0), (0, pad_1), (0, pad_2)), mode="constant", constant_values=0)

    return arr.rechunk(chunks=(1, arr.shape[1], arr.shape[2]))

def resize_arr_to_original_size(arr: da.Array, path, shape_limit = (640, 640)):
    # get original shapes...
    orig = tiff_to_array(path)
    if orig.ndim == 3:
        Z_SIZE, Y_SIZE, X_SIZE = orig.shape
    
    # check last two dims of data and get them to the shape limit
    if(orig.shape[-2] > shape_limit[0] or orig.shape[-1] > shape_limit[1]):
        max_shape = np.max([arr.shape[-2], orig.shape[-1]])
        limit = int(np.ceil(max_shape/32)*32 ) # upper multiple of 32
        shape_limit = (limit, limit)
    
    pad_0 = (shape_limit[0] - orig.shape[-2]) // 2
    pad_1 = (shape_limit[1] - orig.shape[-1]) // 2

    print("Pad vals ", pad_0, pad_1)
    print("B", arr.shape)
    # crop back 
    arr = arr[:, pad_0:-pad_0, pad_1:-pad_1]
    # arr = da.pad(orig, ((0,0), (pad_0, pad_0), (pad_1,pad_1)), mode="constant", constant_values = 0)
    print("A", arr.shape)
    pad_1, pad_2 = 0, 0
    if orig.shape[-2] % 2 != 0:
        pad_1 = 1
    if orig.shape[-1] % 2 != 0:
        pad_2 = 1
    print("Pad vals 2", pad_1, pad_2)

    if pad_1:
        arr = arr[:, :-pad_1, :]
    if pad_2:
        arr = arr[:, :, :-pad_2]
    
    # arr = arr[:, :-pad_1, :-pad_2]
    print("A 2", arr.shape)
    # reshape back to (t, z, y, x) format
    arr = arr.reshape(Z_SIZE, arr.shape[1], arr.shape[2]) # crop back down?

    return arr.astype(orig.dtype)

def segmentation_pipeline(data: da.Array, model: SegmentationModel, batch_size: int = 1) -> da.Array:

    # pre-allocate
    pred = da.zeros_like(data)

    progress_bar = tqdm(range(0, data.shape[0], batch_size))

    for i in progress_bar:
        
        progress_bar.set_description(f"Inference: ")

        # model inference
        arr = np.array(data[i:i+batch_size, :, :]) # why is this nessecary (chunk issue?)        
        mask = model.inference(arr)

        # save output array
        pred[i:i+batch_size, :, :] = mask.squeeze(1)

    return pred

import time
def main(checkpoint_path, data_path):

    data = load_2d_data(data_path)
    
    model = SegmentationModel(checkpoint_path)

    pred = segmentation_pipeline(data, model)
    pred = resize_arr_to_original_size(pred, data_path)
    time_str = time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime())
    tifffile.imwrite(f"{time_str}_pred.tif", pred.compute())

if __name__ == "__main__":
    main("./model_radius_sqrt_new_train.pt", "E:/Sudipta/Arpan/ML_Data/data_slice_10")
