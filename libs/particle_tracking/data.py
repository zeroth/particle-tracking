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

class Data2D:
    def __init__(self, img_dir_path:Path, labels_dir_path:Path) -> None:
        self.img_dir_path = img_dir_path
        self.lables_dir_path = labels_dir_path
        self.org_images = self.load_data(self.img_dir_path)
        self.org_labels = self.load_data(self.lables_dir_path)
        self.images = self.prepare_data_for_ml(self.org_images)
        self.labels = self.prepare_data_for_ml(self.org_labels)
    
    def load_data(self, data_path:Path):
        # return imread(os.path.join(data_path,"*.tif"))
        return imread(data_path)
    
    def prepare_data_for_ml(self, data:da.Array, shape_limit=(640,640)):
        if data.ndim < 2:
            raise Exception("Data has to be minumum two dimintional you have provided only one")
        
        # shape needs to be divisible by 32
        # the model downsamples the images by factor of 2x each layer pass, so needs to have the dimensions match that.
        # 640x640 isnt the upper limit, its just the default shape. chosen arbitrarily because that is the data we started with

        # check last two dims of data and get them to the shape limit
        if(data.shape[-2] > shape_limit[0] or data.shape[-1] > shape_limit[1]):
            max_shape = np.max([data.shape[-2], data.shape[-1]])
            limit = int(np.ceil(max_shape/32)*32 ) # upper multiple of 32
            shape_limit = (limit, limit)
        
        pad_0 = (shape_limit[0] - data.shape[-2]) // 2
        pad_1 = (shape_limit[1] - data.shape[-1]) // 2

        data = da.pad(data, ((0,0), (pad_0, pad_0), (pad_1,pad_1)), mode="constant", constant_values = 0)
        pad_1, pad_2 = 0, 0
        if data.shape[-2] % 2 != 0:
            pad_1 = 1
        if data.shape[-1] % 2 != 0:
            pad_2 = 1
        
        data = da.pad(data, ((0, 0), (0, pad_1), (0, pad_2)), mode="constant", constant_values=0)

        return data.rechunk(chunks=(1, data.shape[1], data.shape[2]))
    
    def resize_arr_to_original_size(self, data: da.Array, path:Path, shape_limit = (640, 640)):
        # get original shapes...
        orig = self.load_data(path)
        if orig.ndim == 3:
            Z_SIZE, Y_SIZE, X_SIZE = orig.shape
        
        # check last two dims of data and get them to the shape limit
        if(orig.shape[-2] > shape_limit[0] or orig.shape[-1] > shape_limit[1]):
            max_shape = np.max([data.shape[-2], orig.shape[-1]])
            limit = int(np.ceil(max_shape/32)*32 ) # upper multiple of 32
            shape_limit = (limit, limit)
        
        pad_0 = (shape_limit[0] - orig.shape[-2]) // 2
        pad_1 = (shape_limit[1] - orig.shape[-1]) // 2

        print("Pad vals ", pad_0, pad_1)
        print("B", data.shape)
        # crop back 
        data = data[:, pad_0:-pad_0, pad_1:-pad_1]
        # arr = da.pad(orig, ((0,0), (pad_0, pad_0), (pad_1,pad_1)), mode="constant", constant_values = 0)
        print("A", data.shape)
        pad_1, pad_2 = 0, 0
        if orig.shape[-2] % 2 != 0:
            pad_1 = 1
        if orig.shape[-1] % 2 != 0:
            pad_2 = 1
        print("Pad vals 2", pad_1, pad_2)

        if pad_1:
            data = data[:, :-pad_1, :]
        if pad_2:
            data = data[:, :, :-pad_2]
        
        # arr = arr[:, :-pad_1, :-pad_2]
        print("A 2", data.shape)
        # reshape back to (t, z, y, x) format
        data = data.reshape(Z_SIZE, data.shape[1], data.shape[2]) # crop back down?

        return data.astype(orig.dtype)
    

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