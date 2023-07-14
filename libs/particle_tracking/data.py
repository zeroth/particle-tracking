import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import tifffile
import dask.array as da
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from .utils import reshape_input_data, revert_tensor_shape, reshape_output_data, convert_tensor_shape, load_tiff

class Data2D:
    def __init__(self, image:da.Array, labels:da.Array) -> None:
        self.org_images = image
        self.org_labels = labels
        self.images = reshape_input_data(self.org_images)
        self.labels = reshape_input_data(self.org_labels)
    
    @staticmethod
    def create(img_dir_path:Path, labels_dir_path:Path):
        org_images = load_tiff(img_dir_path)
        org_labels = load_tiff(labels_dir_path)
        return Data2D(image=org_images, labels=org_labels)


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

        train_dataset = Subset(dataset=self, indices=train_indices)
        test_dataset = Subset(dataset=self, indices=val_indices)

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

        return train_dataloader, test_dataloader

    def get_single_data_loader(self, batch_size = 1):
        dataloader = DataLoader(dataset=self, batch_size=batch_size,
            num_workers=1,
            shuffle=True,
            pin_memory=False)
        return dataloader