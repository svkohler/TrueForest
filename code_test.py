import os
from box import Box
import rioxarray as rxr
import earthpy as et

import rasterio
from rasterio.plot import show as sh
from torchvision.transforms.transforms import ToPILImage
import yaml
import torch

import torchvision.models as models

from PIL import Image
import torch
from torch import nn, optim
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn.functional as F

import numpy as np


# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))

# print(model_names)


# img = Image.open(
#     '/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data/satellite_rgb/satellite_0.png')

# img.show()

# img_satellite = ToTensor()(img)

# # blurrer = transforms.GaussianBlur(kernel_size=[23, 23], sigma=(0.1, 2.0))

# #img_satellite = blurrer(img_satellite)

# img_satellite = transforms.functional.adjust_saturation(img_satellite, 0)

# img = ToPILImage()(img_satellite)

# img.show()

a = np.array([[1, 1], [2, 2], [3, 3]])
print(a)
b = a[1:, :]
print(b)
b = np.append(b, [a[0, :]], axis=0)

print(b)
