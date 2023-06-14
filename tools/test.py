import dask.array as da
import os
import tifffile
from torch.utils.data import DataLoader
# path = "D:/Data/Sudipta/Arpan/ML/Data/exp1"
# arr = da.from_zarr(os.path.join(path, "ord_data.zarr"))
# os.makedirs(os.path.join(path, "org"), exist_ok=True)
# for i in range(arr.shape[0]):
#     tifffile.imwrite(os.path.join(path, "org",f"org_{i}.tiff"), arr[i])

import numpy as np
from multiprocessing import Pool
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import tifffile
from dask_image.imread import imread
import dask.array as da
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import logging
# from .utils import reshape_input_data, revert_tensor_shape, reshape_output_data, convert_tensor_shape

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
    return da.pad(input_data, padding, 'constant')

def reshape_output_data(input_data:da.Array, ouput_data:da.Array):
    padding = _get_padding(input_data, 32)
    w_pad = padding[-1]
    h_pad = padding[-2]
    slices = [[0,0],] * ouput_data.ndim
    slices[-1] = [w_pad[0], -w_pad[1]]
    slices[-2] = [h_pad[0], -h_pad[1]]
    if ouput_data.ndim == 2:
        # h & w
        return ouput_data[slices[-2][0]:slices[-2][1], slices[-1][0]:slices[-1][1]]
    elif ouput_data.ndim == 3:
        # z/c, h & w
        return ouput_data[:, slices[-2][0]:slices[-2][1], slices[-1][0]:slices[-1][1]]
    else:
        # t/b, z/c, h & w
        return ouput_data[:, :, slices[-2][0]:slices[-2][1], slices[-1][0]:slices[-1][1]]


class Data2D:
    def __init__(self, image:da.Array, labels:da.Array) -> None:
        self.org_images = image
        self.org_labels = labels
        self.images = Data2D.prepare_data_for_ml(self.org_images)
        self.labels = Data2D.prepare_data_for_ml(self.org_labels)
    
    @staticmethod
    def load_tiff(data_path:Path):
        # return imread(os.path.join(data_path,"*.tif"))
        return imread(data_path)
    
    @staticmethod
    def create(img_dir_path:Path, labels_dir_path:Path):
        org_images = Data2D.load_tiff(img_dir_path)
        org_labels = Data2D.load_tiff(labels_dir_path)
        return Data2D(image=org_images, labels=org_labels)
    
    @staticmethod
    def prepare_data_for_ml(data:da.Array):
        data = reshape_input_data(data)
        # data = convert_tensor_shape(data)
        return data.rechunk(chunks=(1, data.shape[-2], data.shape[-1]))
    
    @staticmethod
    def resize_arr_to_original_size(org_images, data: da.Array, path:Path, shape_limit = (640, 640)):
        # data = revert_tensor_shape(org_images)

        data = reshape_output_data(org_images, data)

        return data.astype(org_images.dtype)
    

class Dataset2D(Dataset):
    def __init__(self, images, labels, transform=transforms.Compose([transforms.ToTensor()])):
        self.data = images
        self.labels = labels
        self.transform = transform #transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.asarray(self.data[idx], dtype=np.float32)
        mask = np.asarray(self.labels[idx], dtype=np.float32)
        
        # make sure the mask is binary
        mask = np.where(mask > 0, 1, 0)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return (image, mask)
    
    def get_data_loader(self, split, batch_size = 1):
        dataset_size = len(self)
        indices = list(range(dataset_size))
        split = int(np.floor(split * dataset_size))

        np.random.seed(42) # random_seed
        np.random.shuffle(indices)

        val_indices, train_indices = indices[split:], indices[:split]
        val_acc_split = int(len(val_indices)/2)
        val_indices, acc_indices = val_indices[val_acc_split:], val_indices[:val_acc_split]

        train_dataset = Subset(dataset=self, indices=train_indices)
        test_dataset = Subset(dataset=self, indices=val_indices)
        acc_dataset = Subset(dataset=self, indices=acc_indices)

        # n_workers = int(os.cpu_count()/2) if os.cpu_count() >=8 else 4
        n_workers = 1
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=True,
            pin_memory=False
        )

        test_dataloader = DataLoader(
            dataset= test_dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=False,
            pin_memory=False
        )

        acc_dataloader = DataLoader(
            dataset= acc_dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=False,
            pin_memory=False
        )

        return train_dataloader, test_dataloader, acc_dataloader



def main():
    # root = "."
    # SimpleOxfordPetDataset.download(root)
    # train_dataset = SimpleOxfordPetDataset(root, "train")
    # print(train_dataset[0]['image'].shape)
    # n_cpu = os.cpu_count()
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    # print(f"Train size: {len(train_dataset)}")
    # for batch in train_dataloader:
    #     print(batch['image'].shape)
    logging.basicConfig(format="%(asctime)s: %(funcName)s -> %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    logging.info("Test")
    images = Data2D.load_tiff("E:/Sudipta/Arpan/ML_Data/data/send-1.tif")
    labels = Data2D.load_tiff("E:/Sudipta/Arpan/ML_Data/label/mask_label_sqrt_r.tif")
    data = Data2D(image=images, labels=labels)

    dataset = Dataset2D(data.images, data.labels)
    train_dataloader, test_dataloader, acc_dataloader = dataset.get_data_loader(split=0.7)
    for image, mask in test_dataloader:
        print("image.shape", image.shape)
        print("mask.shape", mask.shape)

    

if __name__ == '__main__':
    main()

