from pathlib import Path
import random
import dask.array as da
import numpy as np
import torch

from dask_image.imread import imread
import pandas as pd
import trackpy
from skimage.measure import label, regionprops_table
from tqdm import tqdm

def load_tiff(data_path:Path):
    # return imread(os.path.join(data_path,"*.tif"))
    data = imread(data_path)
    return data.astype(np.float32).rechunk(chunks=(1, data.shape[-2], data.shape[-1]))

def iou(prediction:torch.Tensor, mask:torch.Tensor, eps:float = -1e-7):
    intersection = torch.sum(mask * prediction)
    union = torch.sum(mask) + torch.sum(prediction) - intersection + eps
    return (intersection+eps)/union

def accuracy(prediction:torch.Tensor, mask:torch.Tensor):
    tp = torch.sum(mask == prediction, dtype=prediction.dtype)
    score = tp/mask.view(-1).shape[0]
    return score

def float_to_unit8(tensor:torch.Tensor):
    tensor = torch.sigmoid(tensor)
    tensor = (tensor>0.5).float()
    tensor = tensor * 255.0
    return tensor

def seed_everything(seed:int = 42):
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    da.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# array shape related helper function for 2d data and dask array 
def _get_padding(x:da.Array, stride:int=32):
    # new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
    # new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
    h, w = x.shape[-2:]
    s = stride
    w_padding = int((s-w%s)/2) if w%s >0 else 0
    h_padding = int((s-h%s)/2) if h%s >0 else 0
    w_padding_offset = int((s-w%s) % 2)
    h_padding_offset = int((s-h%s) % 2)
    padding = [(0,0),] * x.ndim
    padding[-1] = (w_padding, w_padding+w_padding_offset)
    padding[-2] = (h_padding, h_padding+h_padding_offset)
    return padding

def convert_tensor_shape(x:da.Array):
    # make sure image has four dimentions (b,c,w,h)
    while len(x.shape) < 4:
        x = da.expand_dims(x, 0)

    # make sure that iteration axis is always at 0th pos
    # eg. if data is t,h,w then it will be t, c, h, w. where t is time
    # eg. if data is z,h,w then it will be z, c, h, w. where z is time or iteration axis
    return da.transpose(x, (1,0,2,3))

def revert_tensor_shape(x:da.Array):
    return da.squeeze(x)

def reshape_input_data(input_data:da.Array):
    # Unet requires the data to be of shape divisible by 32
    padding = _get_padding(input_data, 32)
    data =  da.pad(input_data, padding, 'constant')
    return data.rechunk(chunks=(1, data.shape[-2], data.shape[-1]))

def reshape_output_data(org_data:da.Array, ouput_data:da.Array):
    padding = _get_padding(org_data, 32)
    w_pad = padding[-1]
    h_pad = padding[-2]
    slices = [[0,0],] * ouput_data.ndim
    slices[-1] = [w_pad[0], -w_pad[1]]
    slices[-2] = [h_pad[0], -h_pad[1]]
    data = ouput_data
    if ouput_data.ndim == 2:
        # h & w
        data = ouput_data[slices[-2][0]:slices[-2][1], slices[-1][0]:slices[-1][1]]
    elif ouput_data.ndim == 3:
        # z/c, h & w
        data = ouput_data[:, slices[-2][0]:slices[-2][1], slices[-1][0]:slices[-1][1]]
    else:
        # t/b, z/c, h & w
        data = ouput_data[:, :, slices[-2][0]:slices[-2][1], slices[-1][0]:slices[-1][1]]
    
    return data.astype(org_data.dtype)

# mask position
def get_frame_position_properties(frame:int, mask:np.ndarray, image:np.ndarray=None, result:pd.DataFrame=None) -> pd.DataFrame:
    mask_label = label(mask)
    properties_keys = ['label','centroid', 'intensity_mean', 'intensity_max', 'intensity_min', 'area']
    properties = regionprops_table(label_image=mask_label, intensity_image=image, properties=properties_keys)
    pf = pd.DataFrame(properties)
    pf['frame'] = frame

    if result is None:
        result= pf
    else:
        result = pd.concat([result, pf], ignore_index =True)
    
    return result


def get_statck_properties(masks:np.ndarray, images:np.ndarray, result:pd.DataFrame=None, show_progress=False) -> pd.DataFrame:
    assert images.shape == masks.shape

    iter_range = tqdm(range(images.shape[0])) if show_progress else range(images.shape[0])

    for i in iter_range:
        image = images[i]
        mask = masks[i]
        result = get_frame_position_properties(frame=i, mask=mask, image=image, result=result)
    result.rename(columns={'centroid-0':'y',  'centroid-1':'x'}, inplace=True)
    return result

def get_tracks(df:pd.DataFrame, search_range:float=2, memory:int=0, show_progress:bool=False) -> pd.DataFrame:
    trackpy.quiet((not show_progress))
    return trackpy.link(f=df,search_range=2, memory=0)

def test_shapes():
    img = np.ones((10, 138,181))
    print("img", img.shape)
    
    padded_img = reshape_input_data(img)
    print("padded_img", padded_img.shape)
    
    unpadded_img = reshape_output_data(img, padded_img)
    print("unpadded_img", unpadded_img.shape)

    tensor_shape_img = convert_tensor_shape(padded_img)
    print("tensore_shape_img", tensor_shape_img.shape)

    revert_tensor_shape = revert_tensor_shape(padded_img)
    print("revert_tensor_shape", revert_tensor_shape.shape)

if __name__ == "__main__":
    test_shapes()
