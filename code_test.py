import os
from box import Box
import rioxarray as rxr
import earthpy as et

import rasterio
from rasterio.plot import show as sh
import yaml

config = Box.from_yaml(
    filename="/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForrest/configs/config.yaml")

print(config)
