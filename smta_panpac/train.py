import tqdm
import torch
import time


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

    for e in range(0, epochs):
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

    if savepath:
        print('Saving model at: {savepath}')
        torch.save(model, savepath)

    print('Complete')
    return model, history
