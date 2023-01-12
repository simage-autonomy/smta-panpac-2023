import os, sys
import tqdm
import argparse
import numpy as np
import torch

OUTPUT_DIR = './outputs/'
TRAIN_DIR = '/mnt/c/Users/c_yak/Downloads/smta-data/Morning_Off_Off'

def train_vcnn():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help='Output directory.', default=OUTPUT_DIR)
    parser.add_argument('--train_dir', help='Training directory.', default=TRAIN_DIR)

    args = parser.parse_args()

    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor
    from torch.optim import Adam
    from torch import nn
    import torch
    import time

    from smta_panpac.models import VanillaCNN
    from smta_panpac.data import AirsimDataset
    from smta_panpac.train import train

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

    '''

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
