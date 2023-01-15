import os
import pandas as pd
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchvision.io import read_image, ImageReadMode


class AirsimDataset(Dataset):
    def __init__(self, img_dir, annotations_file='airsim_rec.txt', transform=None, target_transform=None, start=300, end=300):
        self.annotations = pd.read_csv(os.path.join(img_dir, annotations_file), delimiter='\t')
        # Drop the first and last 300 images because these are when the drone takes off and lands
        self.annotations = self.annotations[start:-end].reset_index(drop=True)
        self.img_dir = os.path.join(img_dir, 'images')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.loc[idx, 'ImageFile'])
        try:
            image = read_image(img_path, mode=ImageReadMode.RGB)
        except Exception as e:
            print(f'Issue with image at: {img_path}')
            raise e
        # Cast to float tensor
        image = image.to(torch.float)
        pos = np.array([
                float(self.annotations.loc[idx, 'POS_X']),
                float(self.annotations.loc[idx, 'POS_Y']),
                ])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            pos = self.target_transform(pos)
        return image, pos

    @property
    def targets(self):
        t = []
        for idx, an in self.annotations.iterrows():
            pos = np.array(
                    [
                        float(self.annotations.loc[idx, 'POS_X']),
                        float(self.annotations.loc[idx, 'POS_Y']),
                        ]
                    )
            t.append(pos)
        return np.array(t)

