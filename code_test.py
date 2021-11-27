import os
import rioxarray as rxr
import earthpy as et

import rasterio
from rasterio.plot import show as sh

img = rasterio.open(
    '/home/svkohler/OneDrive/Desktop/Masterthesis/Data/Sentinel_SR_image_2018.tif')
sh(img.read([1, 2, 3, 4]))


print(img.height, img.width, img.count)
