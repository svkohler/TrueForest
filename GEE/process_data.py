# Imports
from box import Box
import numpy as np
from PIL import Image
import os
import sys
import scipy
import scipy.ndimage
from skimage import io
import yaml
import argparse
from tqdm import tqdm
from tif_to_png import tif2png

parser = argparse.ArgumentParser()
parser.add_argument('--patch_size',
                    default=200,
                    help='Choose patch size for images in meters'
                    )
args = parser.parse_args()

# load config
try:
    config = Box.from_yaml(
        filename="/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/configs/custom.yaml")
except:
    raise OSError("Does not exist")

NAIP_resolution = config.NAIP_resolution
Sentinel_resolution = config.Sentinel_resolution

# patch size in meters
patch_size = int(args.patch_size)

# load images
Image.MAX_IMAGE_PIXELS = 9999999999

# get the list of naip, sentinel_RGB, sentinel_NIR image patches
naip_patches = os.listdir(config.data_store + '/raw/NAIP')
sentinel_RGB_patches = os.listdir(config.data_store + '/raw/Sentinel_RGB')
sentinel_NIR_patches = os.listdir(config.data_store + '/raw/Sentinel_NIR')
# extract coordinates from image path to later match against
sentinel_RGB_coordinates = [
    patch[19:len(patch)-4] for patch in sentinel_RGB_patches]
sentinel_NIR_coordinates = [
    patch[19:len(patch)-4] for patch in sentinel_NIR_patches]

# loop over naip image patches and search for the corresponding sentinel satellite patch
converter = tif2png()
for patch in naip_patches:
    # get naip coordinated
    coordinates = patch[11:len(patch)-4]
    # find the indices of the corresponding sentinel images
    idx_RGB = sentinel_RGB_coordinates.index(coordinates)
    idx_NIR = sentinel_NIR_coordinates.index(coordinates)

    # read the images
    naip = Image.fromarray(
        np.uint8(io.imread(config.data_store + '/raw/NAIP/'+patch))[:, :, :3])
    sentinel_rgb = Image.fromarray(np.uint8(io.imread(
        config.data_store + '/raw/Sentinel_RGB/'+sentinel_RGB_patches[idx_RGB])))
    sentinel_nir = Image.fromarray(np.uint8(io.imread(
        config.data_store + '/raw/Sentinel_NIR/'+sentinel_NIR_patches[idx_NIR])))

    print('Loaded coordiantes: ', coordinates)

    converter(config, naip, sentinel_rgb, sentinel_nir, patch_size,
              NAIP_resolution, Sentinel_resolution)

    print('Succesfully cropped coordiantes: ', coordinates)
