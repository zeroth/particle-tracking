import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch import nn, optim
from tqdm import tqdm
from pprint import pprint
import logging
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import dask.array as da
import tifffile
# import zarr
from torch.utils.data.sampler import SubsetRandomSampler


def get_dataloaders(dataset: Dataset, split: float = 0.7, batch_size=32) -> tuple:
    """Randomly split dataset into train/test and generate dataloaders."""
    
    # filter out proportion of empty masks
    # dataset = smart_filtering(dataset,ratio)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split * dataset_size))

    np.random.seed(42) # random_seed
    np.random.shuffle(indices)
    val_indices, train_indices = indices[split:], indices[:split]
    # train_dataset_sampler = SubsetRandomSampler(train_indices)
    # test_dataset_sampler = SubsetRandomSampler(val_indices)
    train_dataset = Subset(dataset=dataset, indices=train_indices)
    test_dataset = Subset(dataset=dataset, indices=val_indices)

    # print(f"Filtered dataset length: {len(dataset)}")

    # indices = torch.randperm(len(dataset)).tolist()
    # idx = int(split * len(indices))

    # train_dataset = Subset(dataset, indices[:idx])
    # print(f"There are {len(train_dataset)} training images.")

    # test_dataset = Subset(dataset, indices[idx:])
    # print(f"There are {len(test_dataset)} test images.")
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


def tiff_to_array(path):
    arr = tifffile.imread(os.path.join(path, "*.tif*"))
    return arr

def load_2d_data(path, shape_limit = (640, 640)):
    arr = tiff_to_array(path)

    if arr.ndim < 2:
        raise Exception("Data has to be minumum two dimintional you have provided only one")
    
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

def train_one_epoch(model, dataloader, device, optimizer):
    logging.info("Training epoc pogress")
    for i, (image, mask) in enumerate(tqdm(dataloader, leave=False)):
        # print(i, sep=", ", end=" ")
        image = image.to(device)
        mask = mask.to(device)

        output = model(image)
        bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()
        dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
        focal_loss_fn = smp.losses.FocalLoss(mode="binary" )

        c_loss = bce_loss_fn(output.float(), mask.float()) 
        d_loss = dice_loss_fn(output.float(), mask.float())
        f_loss = focal_loss_fn(output.float(), mask.float())
        loss = c_loss + d_loss + f_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def iou_v2(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Get IOU between two numpy arrays
    """
    sanity = 1e-7
    intersection = torch.sum(torch.logical_and(outputs, labels))
    union = torch.sum(torch.logical_or(outputs, labels))
    iou = (intersection + sanity) / (union + sanity)
    return iou

def validate_one_epoch(dataloader, model, device: torch.device, loss_fn):
    """Validate an epoch, get iou and dice scores.
    
    
    Returns:
        tuple: (iou, dice): IOU and DICE score for dataset
    """
    model.eval()
    
    iou_net = 0
    dice_score = 0
    loss_f = 0
    
    for i, (img, mask) in enumerate(tqdm(dataloader, leave=False)):
        with torch.no_grad():
            
            # inference
            img = img.to(device) 
            mask = mask.to(device).float()
            pred = model(img).float()
            
            # loss
            loss = loss_fn(pred, mask)
            loss_f += loss.item()
            
            # iou
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float().detach()
            curr_iou = iou_v2(pred, mask).item()
            iou_net += curr_iou
            
            # dice
            dice_c = (2 * torch.sum(pred * mask)) / (torch.sum(pred + mask) + 1e-12)
            dice_score += dice_c.item()

    return (iou_net / (len(dataloader)), dice_score / (len(dataloader)))

class TwoDDataset(Dataset):
    def __init__(self, raw_data_path, labels_data_path, transform=None):
        self.data = load_2d_data(raw_data_path)
        self.labels = load_2d_data(labels_data_path)
        self.transform = transforms.Compose([transforms.ToTensor()])

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

def multi_loss_fn(pred: torch.Tensor, target: torch.Tensor):
    bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()
    dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
    focal_loss_fn = smp.losses.FocalLoss(mode="binary" )

    c_loss = bce_loss_fn(pred, target) 
    d_loss = dice_loss_fn(pred, target)
    f_loss = focal_loss_fn(pred, target)
    loss = c_loss + d_loss + f_loss

    # print(f"bce: {c_loss.item():.4f}, dice: {d_loss.item():.4f}, focal: {f_loss.item():.4f}, sum: {loss.item():.4f}")

    return loss

def train(dataloader, model, optimizer, loss_fn, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (image, mask) in enumerate(dataloader):
        image, mask = image.to(device), mask.to(device)
        output = model(image)
        # bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()
        # dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
        # focal_loss_fn = smp.losses.FocalLoss(mode="binary" )

        # c_loss = bce_loss_fn(output.float(), mask.float()) 
        # d_loss = dice_loss_fn(output.float(), mask.float())
        # f_loss = focal_loss_fn(output.float(), mask.float())
        # loss = c_loss + d_loss + f_loss
        loss = loss_fn(output.float(), mask.float())

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(f"Current batch : {batch}, image len: {len(image)}, image type: {type(image)}")
        if batch % 100 == 0:
            loss_, current = loss.item(), (batch+1)* len(image)
            print(f"loss: {loss_:>7f} [{current:>5d}/{size:>5d}], batch: {batch}")


def iou_dice(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Get IOU between two numpy arrays
    """

    pred = torch.where(outputs < 0.5, 0, 1)

    intersection = torch.sum(pred* labels)
    # print(f"intersection: {intersection}")
    union = torch.sum(pred) + torch.sum(labels)

    # print(f"union: {union}")
    iou = (intersection+1.0) / (union - intersection +1.0)

    dice = (2*intersection)/union
    return (iou, dice)

from torcheval.metrics import BinaryAccuracy
from torcheval.metrics.functional import binary_accuracy
def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batchs = len(dataloader)
    # print(f"num_batchs {num_batchs}, size: {size}")
    model.eval()
    test_loss, correct = 0.0, 0.0
    dice_score, iou_score = 0.0, 0.0
    with torch.no_grad():
        for image, mask in dataloader:
            # print(image.ndim, mask.ndim)
            image, mask = image.to(device), mask.to(device).float()
            pred = model(image)
            test_loss += loss_fn(pred, mask).item()
            
            iou, dice = iou_dice(pred.flatten(), mask.flatten())
            iou_score += iou.item()
            dice_score += dice.item()
            
            correct += binary_accuracy(pred.flatten(), mask.flatten()).item()
            
    test_loss /= num_batchs
    iou_score /= size
    dice_score /= size
    correct /= size
    print(f"Avg loss: {test_loss:>8f}, Accuracy: {correct:>8f}%, IOU : {iou_score:8f}, Dice: {dice_score:>8f}")

def main(raw_data_path, label_data_path, input_model="./checkpoint.pt", output_model="./checkpoint.pt"):
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")

    logging.info("Started")

    n_epochs = 10
    learning_rate = 1.e-4
    train_test_split = 0.7
    batch_size = 1
    input_model_path = input_model # already pre trained model
    output_model_path = output_model

    raw_data = raw_data_path
    label_data = label_data_path
    

    device = torch.device("cpu")
    if torch.cuda.is_available():
        logging.info("Switching device to Cuda")
        device = torch.device("cuda")
    
    logging.info("Initializing model")

    model = smp.Unet(
        encoder_name='resnet18',
        encoder_weights='imagenet',
        in_channels=1,
        classes=1
    )
    model.to(device)

    logging.info("Get Loss function")
    loss_function  = nn.BCEWithLogitsLoss()
    logging.info("Create optimizer")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logging.info("Checking for existing model")
    if (input_model_path is not None) and os.path.exists(input_model_path):
        logging.info("Loading existing model")
        input_model = torch.load(input_model_path, map_location=device)
        model.load_state_dict(input_model["state_dict"])
        optimizer.load_state_dict(input_model["optimizer"])
        model.eval()
    
    logging.info("Loading dataset")
    dataset = TwoDDataset(raw_data_path=raw_data, labels_data_path=label_data)
    print(dataset[0][0].ndim)
    # return
    assert (
        dataset.data.shape == dataset.labels.shape
    ), "Data and labels have a different shape..."

     # dataloader
    print("train_test_split ", train_test_split)
    train_dataloader, test_dataloader = get_dataloaders(
        dataset, split=train_test_split, batch_size=batch_size)
    
    best_iou, best_dice = 0, 0
    current_iou, current_dice = 0, 0
    epoch_progress = range(n_epochs)
    logging.info("Training started")
    for t in epoch_progress:
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model=model, optimizer=optimizer, loss_fn = multi_loss_fn, device=device)
        test(test_dataloader, model=model, loss_fn=multi_loss_fn, device=device)
    logging.info("Done")
    # torch.save(model.state_dict(), output_model_path)
    checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

    torch.save(checkpoint, output_model_path)

    # for epoch in epoch_progress:
    #     model.train(True)
    #     # train
    #     train_one_epoch(
    #         model, 
    #         train_dataloader, 
    #         device, 
    #         optimizer)
    #     model.train(False)

    #     current_iou, current_dice = validate_one_epoch(
    #             test_dataloader, model, device, loss_function
    #         )

    #     print("epoc ", epoch, "is trained")
    #     # save best checkpoint
    #     if current_iou >= best_iou or current_dice >= best_dice:

    #         checkpoint = {
    #             "state_dict": model.state_dict(),
    #             "optimizer": optimizer.state_dict(),
    #         }

    #         torch.save(checkpoint, output_model_path)
    #         best_iou = current_iou
    #         best_dice = current_dice
    #         # epoch_progress.set_description(
    #         #     f"Best epoch {epoch}(iou={best_iou}, dice={best_dice})"
    #         # )

import argparse
import time
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]..."
    )

    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('raw', nargs='?', const=str)
    parser.add_argument('labels', nargs='?', const=str)
    parser.add_argument('-im','--inputmodel', nargs='?', const=str, default=None)
    parser.add_argument('-om','--outputmodel', nargs='?', const=str)
    
    # time_str = time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime())
    # parser.add_argument('-p', '--prefix', nargs=1, type=str, default=time_str)
    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args("E:/Sudipta/Arpan/ML_Data/data E:/Sudipta/Arpan/ML_Data/label -om ./model_radius_sqrt_new_train.pt".split())
    # args = parser.parse_args("D:/Data/Sudipta/Arpan/ML_Training/data D:/Data/Sudipta/Arpan/ML_Training/label -om ./checkpoint_1.pt")
    print(args)
    labels = args.labels
    raw = args.raw
    input_model = args.inputmodel
    output_model = args.outputmodel
    main(raw_data_path=raw, label_data_path=labels, input_model=input_model, output_model=output_model)

    # labels = "D:/Data/Sudipta/Arpan/ML_Training/label"
    # raw = "D:/Data/Sudipta/Arpan/ML_Training/data"
    # labels = "E:/Sudipta/Arpan/ML_Data/label"
    # raw = "E:/Sudipta/Arpan/ML_Data/data"
    # main(raw, labels, None, "./checkpoint_radius_2.pt")
