import os
import sys
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.transforms import ToTensor


class TrueForrestDataset(Dataset):
    def __init__(self, config, transform=None, target_transform=None):
        self.config = config
        self.satellite_rgb_dir = self.config.data_store + '/satellite_rgb/'
        self.satellite_nir_dir = self.config.data_store + '/satellite_nir/'
        self.drone_dir = self.config.data_store + '/drone/'
        self.len = self.check_len()
        self.satellite_rgb_images = sorted(os.listdir(self.satellite_rgb_dir))
        self.satellite_nir_images = sorted(os.listdir(self.satellite_nir_dir))
        self.drone_images = sorted(os.listdir(self.drone_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_satellite = ToTensor()(Image.open(
            self.satellite_rgb_dir + self.satellite_rgb_images[idx]))
        # augment with near infrared channel if activated
        if self.config.NIR:
            img_satellite_nir = ToTensor()(Image.open(
                self.satellite_nir_dir + self.satellite_nir_images[idx]))
            img_satellite = torch.cat(
                (img_satellite, img_satellite_nir), dim=0)
        img_drone = ToTensor()(Image.open(
            self.drone_dir + self.drone_images[idx]))
        # perform transformations
        if self.transform:
            img_satellite = self.transform(img_satellite)
        if self.target_transform:
            img_drone = self.target_transform(img_drone)

        return img_satellite, img_drone

    def check_len(self):
        try:
            len(os.listdir(self.satellite_rgb_dir)) == len(
                os.listdir(self.drone_dir)) == len(os.listdir(self.satellite_nir_dir))
            return len(os.listdir(self.satellite_rgb_dir))
        except:
            ValueError(
                'There is not the same number of drone and satellite images.')


def check_venv(venv='mt_env'):
    if sys.prefix.split('/')[-1] != venv:
        raise ConnectionError('Not connected to correct virtual environment')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
