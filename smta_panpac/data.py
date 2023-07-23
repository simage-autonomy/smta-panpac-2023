import os
import pandas as pd
import numpy as np
import torch
import tqdm
import h5py
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor
from torchvision.io import read_image, ImageReadMode
from transformers import BeitImageProcessor
from PIL import Image


class AirsimDataset(Dataset):
    def __init__(self, h5_path, experiment, group, transform=None, target_transform=None):
        self.h5_path = h5_path
        self._archive = None
        self.group = group
        self.experiment = experiment
        self.transform = transform
        self.target_transform = target_transform

    @property
    def archive(self):
        if self._archive is None:
            self._archive = h5py.File(self.h5_path, 'r')
        return self._archive

    def __len__(self):
        return len(self.archive[self.group][self.experiment]['labels'])

    def __getitem__(self, idx):
        #img = np.rollaxis(self.archive[self.group][self.experiment]['images'][idx], 2, 0)
        img = Image.fromarray(
                self.archive[self.group][self.experiment]['images'][idx],
                mode="RGB"
                )
        # Convert to tensor
        img = ToTensor()(img)
        # Get positional info
        pos = self.archive[self.group][self.experiment]['labels'][idx]
        pos = pos[1:3]
        if self.transform:
            #im = Image.fromarray(img, mode="RGB")
            img = self.transform(img)
        if self.target_transform:
            pos = self.target_transform(pos)
        return img, pos

    @property
    def targets(self):
        return self.archive[self.group][self.experiment]['labels'][:,1:3]


class BeitAirsimDataset(AirsimDataset):
    def __init__(
            self,
            img_dir,
            annotations_file='airsim_rec.txt',
            target_transform=None,
            start=300,
            end=300,
            do_resize=True,
            size={"height": 256, "width": 256},
            do_center_crop=True,
            crop_size={"height": 224, "width": 224},
            do_normalize=True,
            ):
        super().__init__(img_dir, annotations_file=annotations_file, target_transform=target_transform, start=start, end=end)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.loc[idx, 'ImageFile'])
        with Image.open(img_path) as image:
            img = np.array(image)
        pos = np.array([
                float(self.annotations.loc[idx, 'POS_X']),
                float(self.annotations.loc[idx, 'POS_Y']),
                ])
        if self.target_transform:
            pos = self.target_transform(pos)
        return img, pos


        


