from typing import Optional
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm
import os
import dask.array as da
import logging
from torch import nn, optim
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from .data import Data2D
from .utils import reshape_input_data, revert_tensor_shape, reshape_output_data, convert_tensor_shape, load_tiff, iou, accuracy, float_to_unit8

class Model:
    def __init__(self, 
                 pre_trained_model_path:Path =None,
                 optimizerCls=None,
                 learning_rate = 1.e-4,
                 loss_fn=nn.BCEWithLogitsLoss(),
                 encoder_name="resnet18", 
                 encoder_weights="imagenet",
                 in_channels = 1, 
                 classes=1, 
                 is_inference_mode=False) -> None:
        
        self.is_inference_mode = is_inference_mode
        self.encoder_name=encoder_name
        self.encoder_weights=encoder_weights
        self.in_channels = in_channels
        self.classes=classes
        self.loss_fn = loss_fn
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = None
        if not is_inference_mode:
            self.writer = SummaryWriter('runs/unet_segmentaion_{}'.format(timestamp))

        self.device =  (
                    "cuda"
                    if torch.cuda.is_available()
                    # else "mps"
                    # if torch.backends.mps.is_available()
                    else "cpu"
                )
        # self.model = load_model(self.device, checkpoint)
        self.model = smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels = self.in_channels,
            classes=self.classes
        )
        self.model.to(self.device)

        if optimizerCls is not None:
            self.optimizer = optimizerCls(self.model.parameters(), lr=learning_rate)

        logging.info("Checking existing model")
        if (pre_trained_model_path is not None) and os.path.exists(pre_trained_model_path):
            logging.info("Loading existing model")
            input_model = torch.load(pre_trained_model_path, map_location=self.device)
            self.model.load_state_dict(input_model["state_dict"])
            self.optimizer.load_state_dict(input_model["optimizer"])
            self.model.eval()

        
        if is_inference_mode:
            self.model.eval()
    
    def __str__(self) -> str:
        return f"Device : {self.device} \nInference Mode: {self.is_inference_mode} \nEncoder: {self.encoder_name} \nEncoder Weights: {self.encoder_weights} \nIn Channles: {self.in_channels} \nClasses: {self.classes}"

    def train(self, status:bool=True):
        self.model.train(status)
    
    def flush_writer(self):
        if self.writer:
            self.writer.flush()
    
    def close_writer(self):
        if self.writer:
            self.writer.close()
    
    def add_scalar_writer(self, title:str, vals:dict, time_point:int):
        if self.writer:
            self.writer.add_scalar(title,
                        vals,
                        time_point)
            
    def add_scalars_writer(self, title:str, vals:dict, time_point:int):
        if self.writer:
            self.writer.add_scalars(title,
                        vals,
                        time_point)
    
    def save(self, path):
        checkpoint = {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
        torch.save(checkpoint, path)
        
    def pre_process(self, img: np.ndarray) -> torch.Tensor:
        """Pre-process the image for inference"""
        # update data shape to be divisible by 32
        img = reshape_input_data(img.astype(np.float32))
        
        # make it Batch, Channle, Hight, Width
        img = convert_tensor_shape(img)

        # convert to tensor
        img_t = torch.FloatTensor(img.compute())

        # move to device
        img_t = img_t.to(self.device)

        return img_t

    def prediction(self, image):
        with torch.no_grad():
            return self.model(image)
    
    def inference(self, img: np.ndarray) -> np.ndarray:
        if not self.is_inference_mode:
            self.is_inference_mode = True
            self.model.eval()
        """Run model inference on the input image"""
        with torch.no_grad():
            img_t = self.pre_process(img)

            # inference
            output = self.model(img_t)

            # post-process
            mask = self.post_process_image(img, output)
            
            return mask.astype('uint16')
        
    def post_process_image(self, org:np.ndarray, output: torch.Tensor) -> np.ndarray:
        """Post process model output into binary mask."""
        # out1 = torch.sigmoid(output)
        # out1 = (out1 > 0.5).float()
        # pred = out1 * 255.0
        pred = float_to_unit8(output)

        pred = pred.detach().cpu().numpy()

        pred = revert_tensor_shape(pred)
        # resize to the original shape
        pred = reshape_output_data(org, pred)
        return pred
    
    def train_single_set(self, data:tuple):
        # old_mode = self.model.training
        # if not old_mode:
        #     self.model.train()

        image, mask = data
        image, mask = image.to(self.device), mask.to(self.device)

        output = self.model(image)
        loss = self.loss_fn(output.float(), mask.float())

        # clear gradients
        self.optimizer.zero_grad()
        
        # back propogation
        loss.backward()

        # update parameters
        self.optimizer.step()

        # self.model.train(old_mode)
        return loss.item()
    
    def train_one_epoch(self, dataloader, epoch_index):
        running_tloss = 0
        last_loss = 0
        for i, (image, mask) in enumerate(tqdm(dataloader, leave=False, desc=f"Epoch : {epoch_index}")):
            # print(f"\rTraining batch {i}")
            loss = self.train_single_set((image, mask))
            running_tloss += loss
            tb_x = epoch_index * len(dataloader) + i + 1
            self.add_scalar_writer('Loss/Train', loss, tb_x)
            # if i+1 % 100 == 0:
            #     # TODO: consider that our batch size is 1
            #     last_loss = running_tloss / 100 # loss per batch
            #     # print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     tb_x = epoch_index * len(self.train_dataloader) + i + 1
            #     # self.writer.add_scalar('Loss/train', last_loss, tb_x)
            #     running_tloss = 0.

        # our batch size is 1
        return running_tloss / len(dataloader)

    def validate_training(self, dataloader, epoch_number=0, avg_loss=0):
        running_vloss = 0.0
        running_iou = 0.0
        running_vaccuracy = 0.0
        with torch.no_grad():
            for i, (image, mask) in enumerate(tqdm(dataloader, leave=False, desc=f"Validation Epoch {epoch_number}")):
                image, mask = image.to(self.device), mask.to(self.device).float()
                
                prediction = self.model(image)
                vloss = self.loss_fn(prediction, mask)
                #validation loss
                running_vloss += vloss.item()
                tb_x = epoch_number * len(dataloader) + i + 1
                self.add_scalar_writer('VLoss/Train', vloss.item(), tb_x)

                # Accuracy IOU
                iou = self.intensity_over_union(prediction=prediction, mask=mask)
                accuracy = self.accuracy(prediction=prediction, mask=mask)
                # accuracy = smp.utils.functional.iou(prediction, mask)

                running_iou += iou
                running_vaccuracy += accuracy
                self.add_scalar_writer('VAccuracy_iou/Train', iou.item(), tb_x)
                self.add_scalar_writer('VAccuracy/Train', accuracy.item(), tb_x)

        avg_vloss = running_vloss / len(dataloader)
        avg_iou = running_iou / len(dataloader)
        avg_vaccuracy = running_vaccuracy / len(dataloader)
        # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        
        self.add_scalars_writer('Training Loss vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        self.add_scalars_writer('Training Loss vs. Validation Iou',
                        { 'Loss' : avg_loss, 'IOU' : avg_iou },
                        epoch_number + 1)
        self.add_scalars_writer('Training Loss vs. Validation Accuracy',
                        { 'Loss' : avg_loss, 'Accuracy' : avg_vaccuracy },
                        epoch_number + 1)
        
        self.add_scalars_writer('Validation Loss vs. Validation Accuracy',
                        { 'Loss' : avg_loss, 'IOU' : avg_iou },
                        epoch_number + 1)
        
        return avg_vloss, avg_iou, avg_vaccuracy
    
    def intensity_over_union(self, prediction, mask):
        prediction_sig = float_to_unit8(prediction)
        return iou(prediction_sig, mask)
    
    def accuracy(self, prediction, mask):
        prediction_sig = float_to_unit8(prediction)
        return accuracy(prediction_sig, mask)
