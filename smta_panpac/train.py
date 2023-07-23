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
        num_batches_until_report=10,
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

        # Initialize for logging
        past_batch = 0

        for curr_batch, (x, y) in tqdm.tqdm(enumerate(dataloader), desc=f'Batch', total=len(dataloader)):
            # Send the input to the device
            (x, y) = (x.to(torch.float).to(device), y.to(torch.float).to(device))

            # Forward pass
            pred = model(x)
            loss = loss_fn(pred, y)

            # Report
            if curr_batch - past_batch == num_batches_until_report:
                tqdm.tqdm.write(f'Train Loss: {loss}')
                past_batch = curr_batch

            # Zero out grads, preform backprop and update weights
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Add losses
            total_train_loss += loss

        # Calculate average training loss
        avg_train_loss = total_train_loss / train_steps
        history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        tqdm.tqdm.write(f'Average Training Loss: {avg_train_loss}')

    if savepath:
        print(f'Saving model at: {savepath}')
        torch.save(model, os.path.join(savepath, f'{model.name}.pt'))

    print('Complete')
    return model, history

def train_huggingfaces(
        dataloader,
        feature_extractor,
        x_transform_fn,
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
            # Transform x given the transform fn
            x = x_transform_fn(x)
            # Send the input to the device
            inputs = feature_extractor(x, return_tensors="pt").to(device)
            y = y.to(torch.float).to(device)

            # Forward pass
            pred = model(**inputs).logits
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

def predict_huggingfaces(
        dataloader,
        feature_extractor,
        x_transform_fn,
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
            # Transform x given the transform fn
            x = x_transform_fn(x)
            # Send the input to the device
            inputs = feature_extractor(x, return_tensors="pt").to(device)
            y = y.to(torch.float).to(device)

            # get prediction
            pred = model(**inputs).logits

            preds.append(pred.cpu().numpy())
    return np.squeeze(np.array(preds), axis=1)
