import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.transforms import ToTensor


class TrueForrestDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.satellite_dir = img_dir + '/satellite'
        self.drone_dir = img_dir + '/drone'
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path_satellite = os.path.join(self.satellite_dir, )
        img_path_drone = os.path.join(self.drone_dir, )
        img_satellite = ToTensor()(Image.open(img_path_satellite))
        img_drone = ToTensor()(Image.open(img_path_drone))
        if self.transform:
            img_satellite = self.transform(img_satellite)
        if self.target_transform:
            img_drone = self.target_transform(img_drone)
        return img_satellite, img_drone
