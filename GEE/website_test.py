from rasterio import windows
import rasterio as rio
from itertools import product
import os
from pyproj import Proj, transform
from affine import Affine
import rasterio
from rasterio.windows import Window
import ee
import geemap
import sys
import numpy as np
from PIL import Image
from skimage import io
import pickle


# fname = '/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data/raw/NAIP/Central_Valley/train/NAIP_image_36.45_-120.05.tif'


# in_path = '/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data/raw/NAIP/Central_Valley/train/'
# input_filename = 'NAIP_image_36.45_-120.05.tif'

# out_path = '/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data/website_test/'
# output_filename = 'tile_{}-{}.tif'


# def get_tiles(ds, width=448, height=448):
#     nols, nrows = ds.meta['width'], ds.meta['height']
#     offsets = product(range(0, nols, width), range(0, nrows, height))
#     big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
#     for col_off, row_off in offsets:
#         window = windows.Window(col_off=col_off, row_off=row_off,
#                                 width=width, height=height).intersection(big_window)
#         transform = windows.transform(window, ds.transform)
#         yield window, transform


# with rio.open(os.path.join(in_path, input_filename)) as inds:
#     tile_width, tile_height = 448, 448

#     meta = inds.meta.copy()

#     for window, transform in get_tiles(inds):
#         print(window)
#         meta['transform'] = transform
#         meta['width'], meta['height'] = window.width, window.height
#         outpath = os.path.join(out_path, output_filename.format(
#             int(window.col_off), int(window.row_off)))
#         with rio.open(outpath, 'w', **meta) as outds:
#             outds.write(inds.read(window=window))


# imgs = os.listdir(
#     '/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data/website_test/')


# coordinates_images = {}
# counter = 0
# for img in imgs:
#     dataset = rasterio.open(
#         '/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data/website_test/'+img)
#     coordinates = {}
#     coordinates['left'] = dataset.bounds[0]
#     coordinates['bottom'] = dataset.bounds[1]
#     coordinates['right'] = dataset.bounds[2]
#     coordinates['top'] = dataset.bounds[3]
#     coordinates_images[img] = coordinates
#     pil = Image.fromarray(
#         np.uint8(io.imread('/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data/website_test/'+img))[:, :, :3])
#     pil.save(
#         '/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data/website_test/'+img[:-3]+'png')
#     os.remove(
#         '/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data/website_test/'+img)
#     print(f'done {counter}/{len(imgs)}')
#     counter += 1

# with open('/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data/website_test/coordinates_images.pickle', 'wb') as handle:
#     pickle.dump(coordinates_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data/website_test/coordinates_images.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(b['tile_448-3584.tif'])
