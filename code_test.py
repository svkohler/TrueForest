import os
from box import Box
import rioxarray as rxr
import earthpy as et

import rasterio
from rasterio.plot import show as sh
import yaml
import torch

import torchvision.models as models


# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))

# print(model_names)

ten = torch.zeros(4, 5)

input = torch.tensor([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])

print(input.shape, ten.shape)

ten[::2, :] = input

print(ten)
