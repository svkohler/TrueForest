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

import torchvision.models as models
net = models.resnet101()

print(net)
