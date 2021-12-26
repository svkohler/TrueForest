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


a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

c = torch.cat((a, b), dim=0)

print(c, c.shape)
