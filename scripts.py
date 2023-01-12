import os, sys
import tqdm
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch import nn
import torch
import time

from smta_panpac.models import VanillaCNN
from smta_panpac.data import AirsimDataset
from smta_panpac.train import train, predict

OUTPUT_DIR = './outputs/'
TRAIN_DIR = '/mnt/c/Users/c_yak/Downloads/smta-data/Morning_Off_Off'

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_dir', help='Test Directory')
    parser.add_argument('--model_path', help='Path to pre-trained model.', default=None)
    parser.add_argument('--output_dir', help='Output directory.', default=OUTPUT_DIR)
    parser.add_argument('--train_dir', help='Training directory.', default=TRAIN_DIR)
    parser.add_argument('--epochs', help='Number of training epochs.', default=10, type=int)
    parser.add_argument('--batch_size', help='Batch size to use for training', default=64, type=int)
    parser.add_argument('--lr', help='Learning rate for training.', default=1e-3, type=float)

    return parser.parse_args()

def vcnn_experiment():
    # Parse args
    args = parse()
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_path is None:
        print('No model path passed, will train from scratch.')
        model, history = _train_vcnn(args)
    else:
        model = torch.load(args.model_path)

    # Setup test data
    test_data = AirsimDataset(args.test_dir, transform=None, start=300, end=300)
    test_dataloader = DataLoader(test_data)

    # Get predictions
    preds_df = pd.DataFrame(predict(test_dataloader, model, device), columns=['pred_x', 'pred_y', 'pred_z'])

    # Get Extract Ground Truth
    truth_df = pd.DataFrame(test_data.targets, columns=['x', 'y', 'z'])

    # Combine DFs
    df = pd.concat([truth_df, preds_df], axis=1)
    # Save
    res_filepath = os.path.join(args.output_dir, f'vanillacnn-results-{time.strftime("%Y%m%d-%H%M%S")}.csv')
    print(f'Saving results to: {res_filepath}')
    df.to_csv(res_filepath)
    print('Complete.')
    return

def _train_vcnn(args):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data
    data = AirsimDataset(args.train_dir, transform=None)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    # Training Loop
    train_steps = len(dataloader.dataset) // args.batch_size

    ## Initialize model
    model = VanillaCNN(3)

    ## Initialize optimization and loss
    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    return train(
            dataloader,
            model,
            loss_fn,
            opt,
            train_steps,
            args.epochs,
            device,
            args.output_dir,
            )


def train_vcnn():
    args = parse()
    return _train_vcnn(args)

def remove_corrupted_images():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Directory to analyze.')
    args = parser.parse_args()

    from torchvision.io import read_image, ImageReadMode

    # Load annotations
    annotations = pd.read_csv(os.path.join(args.d, 'airsim_rec.txt'), delimiter='\t')
    img_dir = os.path.join(args.d, 'images')

    # Run through and remove any corrupted images
    bad_indices = []
    bad_an = []
    for idx, an in tqdm.tqdm(annotations.iterrows(), total=len(annotations)):
        img_path = os.path.join(img_dir, an['ImageFile'])
        try:
            image = read_image(img_path, mode=ImageReadMode.RGB)
        except Exception as e:
            print(f'Issue with image at: {img_path} so removing...')
            os.remove(img_path)
            bad_indices.append(idx)
            bad_an.append(an)
            
    new_annotations = annotations.drop(bad_indices)
    new_annotations = new_annotations.reset_index(drop=True)
    new_annotations.to_csv(os.path.join(args.d, 'airsim_rec-new.txt'), sep='\t')
    print('Complete.')

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help='Output directory.', default=OUTPUT_DIR)
    parser.add_argument('--train_dir', help='Training directory.', default=TRAIN_DIR)

    args = parser.parse_args()


    # Hyperparameters
    INIT_LR = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 10

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data
    data = AirsimDataset(args.train_dir, transform=None)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    # Training Loop
    train_steps = len(dataloader.dataset) // BATCH_SIZE

    ## Initialize model
    model = VanillaCNN(3)

    ## Initialize optimization and loss
    opt = Adam(model.parameters(), lr=INIT_LR)
    loss_fn = nn.MSELoss()

    train(
            dataloader,
            model,
            loss_fn,
            opt,
            train_steps,
            EPOCHS,
            device,
            args.output_dir,
            )


    # Initialize history
    history = {
            "train_loss": []
            }

    print('Training network...')

    for e in range(0, EPOCHS):
        # Model to training mode
        model.train()

        # Initialize loss
        total_train_loss = 0

        for (x, y) in tqdm.tqdm(dataloader, desc=f'Epoch: {e}, Batch'):
            # Send the input to the device
            (x, y) = (x.to(torch.float).to(device), y.to(torch.float).to(device))

            # Forward pass
            pred = model(x)
            loss = loss_fn(pred, y)

            # Zero out grads, preform backprop and update weights
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Add losses
            total_train_loss += loss

        # Calculate average training loss
        avg_train_loss = total_train_loss / train_steps
        H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        print(f'EPOCH: {e+1}/{EPOCHS}')
        print('Train Loss: {avg_train_loss}')
    print('Complete')
    '''
