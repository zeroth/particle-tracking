import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import tifffile
import torch
from torch import nn, optim
from tqdm import tqdm
from pprint import pprint
import logging
from particle_tracking.data import Data2D, Dataset2D
from particle_tracking.model import Model
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"


def training_loop(n_epochs, model, dataloader):
    best_vloss = 1_000_000.
    # Training loop
    for epoch in tqdm(range(n_epochs), desc="Training Loop"):
        model.train(True)
        # print(f"Training {epoch}\n")
        avg_loss= model.train_one_epoch(dataloader, epoch_index=epoch)
        model.train(False)

        avg_vloss, avg_iou, avg_accuracy  = model.validate_training(dataloader, epoch_number=epoch, avg_loss=avg_loss)
        # avg_accuracy = model.accuracy_check(epoch_number=epoch)
        model.flush_writer()

        # model.add_sclares_writer("Training Avg VLoss vs Avg Accuracy ", {"VLoss": avg_vloss, "Accuracy": avg_accuracy}, epoch)
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            model.add_scalars_writer("Training  Best Vloss vs Avg VLoss ", {"Best": best_vloss, "Avg": avg_vloss}, epoch)
            best_vloss = avg_vloss
            # model_path = 'model_{}_{}.pt'.format(timestamp, epoch)
            # # torch.save(model.state_dict(), model_path)
            # model.save(os.path.join(output_model_path, model_path))

def accuraccy_loop(model:Model, dataloader):
    for i, (image, mask) in enumerate(dataloader):
        image, mask = image.to(model.device), mask.to(model.device)
        # print("image.shape", image.shape)
        prediction = model.prediction(image)
        # print("prediction.shape",prediction.shape)
        iou = model.intensity_over_union(prediction=prediction, mask=mask)
        accuracy = model.accuracy(prediction=prediction, mask=mask)
        print(f"index: {i}, accuracy: {accuracy}, iou: {iou}")

def main(raw_data_path:Path, label_data_path:Path, output_model:Path, input_model:Path = None, n_epochs:int =10):
    logging.info("Started")
    n_epochs = n_epochs
    learning_rate = 1.e-4
    train_test_split = 0.7
    batch_size = 1
    input_model_path = input_model # already pre trained model
    output_model_path = output_model
    os.makedirs(output_model_path, exist_ok=True)

    logging.info("Create Data")
    data = Data2D.create(raw_data_path, label_data_path)

    logging.info("Create Dataset")
    dataset = Dataset2D(data.images, data.labels)
    logging.info("Get Dataloaders")
    train_dataloader, test_dataloader, acc_dataloader = dataset.get_data_loader(split=train_test_split)

    logging.info("Get Loss function")
    loss_function  = nn.BCEWithLogitsLoss()

    logging.info("Create Model")
    model = Model(pre_trained_model_path=input_model_path, 
                  optimizerCls=optim.Adam, 
                  learning_rate=learning_rate, 
                  loss_fn=loss_function
                  )

    best_vloss = 1_000_000.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logging.info("Start Training Loop")
    training_loop(n_epochs=n_epochs, model=model, dataloader = train_dataloader)
    logging.info("End Training Loop")            
    
    logging.info("Save the model")
    time_str = time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime())
    model.save(os.path.join(output_model_path,f"model_final_{time_str}.pt"))

    logging.info("Accuracy testing")
    accuraccy_loop(model, acc_dataloader)

    model.close_writer()

##########################################################################################################
import argparse
import time
def init_argparse() -> argparse.ArgumentParser:
    logging.basicConfig(format="%(asctime)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
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
    parser.add_argument('-e','--epochs', nargs=1, type=int, default=10)
    
    # time_str = time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime())
    # parser.add_argument('-p', '--prefix', nargs=1, type=str, default=time_str)
    return parser


if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args("E:/Sudipta/Arpan/ML_Data/data/send-1.tif E:/Sudipta/Arpan/ML_Data/label/mask_label_sqrt_r.tif -om ./model_radius_sqrt_new_train".split())
    # args = parser.parse_args("D:/Data/Sudipta/Arpan/ML_Training/data/send-1.tif D:/Data/Sudipta/Arpan/ML_Training/label/mask_radius_sqrt.tif -om ./model_radius_sqrt_new_train".split())
    logging.info(f"Main Args : {args}")
    print("test")
    labels = args.labels
    raw = args.raw
    input_model = args.inputmodel
    output_model = args.outputmodel
    epochs = args.epochs
    # epochs = 1
    main(raw_data_path=raw, label_data_path=labels, input_model=input_model, output_model=output_model, n_epochs=epochs)
