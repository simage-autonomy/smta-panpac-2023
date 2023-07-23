import os, sys
import tqdm
import argparse
import torch
import time
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch import nn
from transformers import AutoConfig, BeitImageProcessor, BeitForImageClassification
from PIL import Image

from smta_panpac.models import VanillaCNN
from smta_panpac.data import AirsimDataset, BeitAirsimDataset
from smta_panpac.train import train, train_huggingfaces, predict, predict_huggingfaces
from smta_panpac.compact_transformers.vit import ViTLite

OUTPUT_DIR = './outputs/'
TRAIN_DIR = '/mnt/c/Users/c_yak/Downloads/smta-data/Morning_Off_Off'

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Hdf5 dataset path.')
    parser.add_argument('--exp', help='Name of experiment to run.')
    parser.add_argument('--set', help='Train or Test.', default='train', type=str)
    parser.add_argument('--model_path', help='Path to pre-trained model.', default=None)
    parser.add_argument('--output_dir', help='Output directory.', default=OUTPUT_DIR)
    parser.add_argument('--train_dir', help='Training directory.', default=TRAIN_DIR)
    parser.add_argument('--epochs', help='Number of training epochs.', default=10, type=int)
    parser.add_argument('--batch_size', help='Batch size to use for training', default=64, type=int)
    parser.add_argument('--lr', help='Learning rate for training.', default=1e-3, type=float)
    parser.add_argument('--script_name', help='Name of the script to run.', default=None, type=str)
    parser.add_argument('--model_class', help='Name of the model class.', default=None)
    return parser.parse_args()

def vcnn_experiments():
    # Parse args
    args = parse()
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    f = h5py.File(args.d, 'r')
    train_exps = list(f['train'].keys())
    test_exps = list(f['test'].keys())
    f.close()

    # Run training for each dataset
    for experiment in train_exps:
        print(f'Training on {experiment}')
        args.exp = experiment
        model, history = _train_vcnn(args)

    # Run Testing
    for experiment in test_exps:
        print(f'Testing on {experiment}')
        # Setup test data
        test_data = AirsimDataset(args.d, experiment, 'test',  transform=None)
        test_dataloader = DataLoader(test_data)

        # Get predictions
        preds = predict(test_dataloader, model, device)
    
        # Export results
        export_predictions(test_data, test_dataloader, preds, args.output_dir)
    print('Complete')
    return

def run_experiment():
    # Parse args
    args = parse()
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    f = h5py.File(args.d, 'r')
    test_exps = list(f['test'].keys())
    f.close()

    if args.model_class == 'vcnn':
        model, history = _train_vcnn(args)
    elif args.model_class == 'vit':
        model, history = _train_vit(args)
    elif args.model_class == 'beit':
        model, history = _train_beit(args)
    else:
        raise ValueError(f'Unknown model class {args.model_class}')

    # Run Testing
    for experiment in test_exps:
        print(f'Testing on {experiment}')
        # Setup test data
        test_data = AirsimDataset(args.d, experiment, 'test',  transform=None)
        test_dataloader = DataLoader(test_data)

        # Get predictions
        preds = predict(test_dataloader, model, device)
    
        # Export results
        export_predictions(test_data, test_dataloader, preds, args.output_dir)
    print('Complete')
    return

def run_predictions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Hdf5 dataset path.')
    parser.add_argument('-m', help='Path to model files.')
    parser.add_argument('-o', help='Output directory and where model files are located.')
    args = parser.parse_args()
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    f = h5py.File(args.d, 'r')
    test_exps = list(f['test'].keys())
    f.close()
    
    # Read model files
    with open(args.m, 'r') as f_:
        model_names = f_.readlines()

    # Specify transforms
    # Setup image transformations which will resize to ViT image size
    vit_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation((0, 180)),
        transforms.Normalize(
            mean=torch.tensor([0.5, 0.5, 0.5]),
            std=torch.tensor([0.5, 0.5, 0.5]),
            ),
        ])
    # Transform fn for BEiT
    x_transform_fn = lambda x: [Image.fromarray(_x, mode="RGB") for _x in x.numpy()]
    feature_extractor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')

    # Run predictions and tests for each model file
    for model_name in model_names:
        model_name = model_name.strip()
        print(f'Predicting with {model_name}.')
        model_path = os.path.join(args.o, model_name)
        model_name_noext = os.path.splitext(model_name)[0]
        model = torch.load(model_path)
        
        # Set the model name
        model.name = model_name_noext

        # Model specific criteria
        if 'vit' in model_name_noext:
            transform = vit_transform
        else:
            transform = None

        # Run Testing
        for experiment in test_exps:
            print(f'Testing on {experiment}')
            # Setup test data
            test_data = AirsimDataset(args.d, experiment, 'test',  transform=transform)
            test_dataloader = DataLoader(test_data)

            # Get predictions
            if 'beit' in model_name_noext:
                preds = predict_huggingfaces(
                        test_dataloader,
                        feature_extractor,
                        x_transform_fn,  
                        model,
                        device,
                        )
            else:
                preds = predict(test_dataloader, model, device)
        
            # Export results
            export_predictions(test_data, test_dataloader, preds, args.o, model)
    print('Complete')
    return


def export_predictions(
        test_data,
        test_dataloader,
        predictions,
        output_dir,
        model,
        ):
    # Setup predictions dataframe
    preds_df = pd.DataFrame(predictions, columns=['x', 'y'])
    preds_df['sample'] = 'prediction'
    preds_df['timepoint'] = np.array([i for i in range(len(preds_df))])

    # Setup truth dataframe
    truth_df = pd.DataFrame(test_data.targets, columns=['x', 'y'])
    truth_df['sample'] = 'true'
    truth_df['timepoint'] = np.array([i for i in range(len(truth_df))])

    # Combine DFs
    df = pd.concat([truth_df, preds_df])
    # Save
    res_filepath = os.path.join(output_dir, f'{model.name}-results-{test_data.experiment}.csv')
    print(f'Saving results to: {res_filepath}')
    df.to_csv(res_filepath)
    return


def _train_vcnn(args):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data
    data = AirsimDataset(args.d, args.exp, args.set, transform=None)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Training Loop
    train_steps = len(dataloader.dataset) // args.batch_size

    ## Initialize model
    model = VanillaCNN(3, name=f'vcnn-{args.exp}-{time.strftime("%Y%m%d-%H%M%S")}')

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

def _train_vit(args):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup image transformations which will resize to ViT image size
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation((0, 180)),
        transforms.Normalize(
            mean=torch.tensor([0.5, 0.5, 0.5]),
            std=torch.tensor([0.5, 0.5, 0.5]),
            ),
        ])

    # Setup data
    data = AirsimDataset(args.d, args.exp, args.set, transform=transform)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Training Loop
    train_steps = len(dataloader.dataset) // args.batch_size

    ## Initialize model
    model = ViTLite(
            num_layers=2,
            num_heads=2,
            mlp_ratio=1,
            embedding_dim=128,
            num_classes=2,
            )

    # Set model name
    model.name = f'vit-2-2-{args.exp}-{time.strftime("%Y%m%d-%H%M%S")}'

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

def _train_beit(args):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data
    data = AirsimDataset(args.d, args.exp, args.set, transform=None)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    # Training Loop
    train_steps = len(dataloader.dataset) // args.batch_size

    ## Initialize model
    feature_extractor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
    model = BeitForImageClassification.from_pretrained(
            'microsoft/beit-base-patch16-224',
            num_labels=2,
            ignore_mismatched_sizes=True,
            )

    # Set model name
    model.name = f'beit-finetune-{args.exp}-{time.strftime("%Y%m%d-%H%M%S")}'

    ## Initialize optimization and loss
    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    x_transform_fn = lambda x: [Image.fromarray(_x, mode="RGB") for _x in x.numpy()]

    return train_huggingfaces(
            dataloader,
            feature_extractor,
            x_transform_fn,
            model,
            loss_fn,
            opt,
            train_steps,
            args.epochs,
            device,
            args.output_dir,
            )

def train_beit():
    args = parse()
    return _train_beit(args)

def train_vit():
    args = parse()
    return _train_vit(args)

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

def extract_experiment(exp_grp, annotations, img_dir, experiment):
    is_dataset_init = False
    img_data = []
    label_data = []
    for idx, an in tqdm.tqdm(annotations.iterrows(), desc=f'Processing {experiment}', total=len(annotations), leave=False):
        img_path = os.path.join(img_dir, an['ImageFile'])
        try:
            with Image.open(img_path) as image:
                image = image.convert('RGB')
                arr = np.array(image)
        except Exception as e:
            tqdm.tqdm.write(f'Issue with image at: {img_path} so not including.')
            continue
        img_data.append(arr)
        label_data.append(np.array(an)[1:-1].astype(float))
    print('Casting to numpy array.')
    img_data = np.array(img_data[300:-300])
    label_data = np.array(label_data[300:-300])
    img_ds = exp_grp.create_dataset('images', img_data.shape, dtype='f')
    label_ds = exp_grp.create_dataset('labels', label_data.shape, dtype='f')
    print('Saving to hdf5 file.')
    img_ds[...] = img_data
    label_ds[...] = label_data
    label_ds.attrs['label_names'] = [s.lower() for s in list(annotations.columns)[1:-1]]
    return

def create_hdf5():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_dir', help='Path to Training directory to analyze.')
    parser.add_argument('-test_dir', help='Path to Testing directory.')
    parser.add_argument('--name', help='Name of the dataset', default='airsim_data.hdf5')
    args = parser.parse_args()

    # Initialize dataset
    f = h5py.File(args.name, mode='w')
    
    # Create training group
    train_grp = f.create_group('train')

    # Process training data
    for experiment in os.listdir(args.train_dir):
        if '.zip' in experiment:
            continue
        if '.tar.gz' in experiment:
            continue
        # Load annotations
        annotations = pd.read_csv(os.path.join(args.train_dir, experiment, 'airsim_rec.txt'), delimiter='\t')
        img_dir = os.path.join(args.train_dir, experiment, 'images')
        # Create experiment group in training group
        exp_grp = train_grp.create_group(experiment.lower())
        extract_experiment(exp_grp, annotations, img_dir, experiment)
    
    # Create testing group
    test_grp = f.create_group('test')
    
    # Process testing data
    for experiment in os.listdir(args.test_dir):
        if '.zip' in experiment:
            continue
        if '.tar.gz' in experiment:
            continue
        # Load annotations
        annotations = pd.read_csv(os.path.join(args.test_dir, experiment, 'airsim_rec.txt'), delimiter='\t')
        img_dir = os.path.join(args.test_dir, experiment, 'images')
        # Create experiment group in training group
        exp_grp = test_grp.create_group(experiment.lower())
        extract_experiment(exp_grp, annotations, img_dir, experiment)

    f.close()
    return

def get_experiment_names():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Hdf5 dataset path.')
    args = parser.parse_args()
    f = h5py.File(args.d, 'r')
    print('Available Training Experiments:')
    for key in f['train'].keys():
        print(f'\t- {key}')
    print('Available Testing Experiments:')
    for key in f['test'].keys():
        print(f'\t- {key}')
    return

if __name__ == '__main__':
    args = parse()

    if not args.script_name:
        raise RuntimeError('Must provide a script name to run.')

    script = globals()[args.script_name]
    script(args)
