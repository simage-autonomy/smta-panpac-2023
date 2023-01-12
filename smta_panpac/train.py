import os
import time
import tqdm
import torch
import time
import numpy as np


def train(
        dataloader,
        model,
        loss_fn,
        opt,
        train_steps,
        epochs,
        device,
        savepath,
        ):
    ## Initialize model
    model = model.to(device)

    # Initialize history
    history = {
            "train_loss": []
            }

    print('Training network...')

    for e in tqdm.tqdm(range(0, epochs), desc='Epoch'):
        # Model to training mode
        model.train()

        # Initialize loss
        total_train_loss = 0

        for (x, y) in tqdm.tqdm(dataloader, desc=f'Batch'):
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
        history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        tqdm.tqdm.write(f'Train Loss: {avg_train_loss}')

    if savepath:
        print(f'Saving model at: {savepath}')
        torch.save(model, os.path.join(savepath, f'{model.name}.pt'))

    print('Complete')
    return model, history

def predict(
        dataloader,
        model,
        device,
        ):

    # Send model to device
    model.to(device)

    print('Evaluating model...')

    with torch.no_grad():
        # Set to eval mode
        model.eval()

        # initialize predictions
        preds = []

        for (x, y) in tqdm.tqdm(dataloader, desc='Evaluating on Test Instance'):
            # Send x to device and cast appropriately
            x = x.to(torch.float).to(device)

            # Get prediction
            pred = model(x)
            preds.append(pred.cpu().numpy())
    return np.squeeze(np.array(preds), axis=1)

