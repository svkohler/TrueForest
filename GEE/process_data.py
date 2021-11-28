# Imports
from box import Box
import numpy as np
from PIL import Image
import os
import scipy
import scipy.ndimage
import yaml
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--patch_size',
                    default=200,
                    help='Choose patch size for images in meters'
                    )
args = parser.parse_args()

# load config
try:
    config = Box.from_yaml(
        filename="/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForrest/configs/config.yaml")
except:
    raise OSError("Does not exist")

NAIP_resolution = config.NAIP_resolution
Sentinel_resolution = config.Sentinel_resolution

# patch size in meters
patch_size = args.patch_size

# load images

sentinel_rgb = Image.open(
    config.data_store + '/raw/Sentinel_SR_image_2018_RGB.tif')
sentinel_nir = Image.open(
    config.data_store + '/raw/Sentinel_SR_image_2018_NIR.tif')
print('Succesfully loaded Sentinel images. Shape: ', sentinel_rgb.size)
# naip = io.imread(config.data_store + 'raw/NAIP_image_2018.tif')
# print('Succesfully loaded NAIP images. Shape: ', naip.shape)

# buckets to collect images
sentinel_crop = []
naip_crop = []

# define function to extract image patches


def tif_images(naip, sentinel, patch_size, res_naip, res_sentinel):
    nr_images = int(naip.size[0]/patch_size)
    patch_size_naip = int(patch_size / res_naip)
    patch_size_sentinel = int(patch_size / res_sentinel)

    if not os.path.exists(config.data_store+'/satellite'):
        os.makedirs(config.data_store+'/satellite')

    if not os.path.exists(config.data_store+'/drone'):
        os.makedirs(config.data_store+'/drone')

    for i in tqdm(range(0, nr_images)):

        naip_crop = naip.crop((i*patch_size_naip, i*patch_size_naip,
                               patch_size_naip + i*patch_size_naip, patch_size_naip+i*patch_size_naip))

        sentinel_crop = sentinel.crop((i*patch_size_sentinel, i*patch_size_sentinel,
                                       patch_size_sentinel+i * patch_size_sentinel, patch_size_sentinel+i*patch_size_sentinel)).resize((patch_size, patch_size))

        sentinel_crop.save(config.data_store+'/satellite/satellite_' +
                           str(i)+'.png')
        naip_crop.save(config.data_store+'/drone/drone_'+str(i)+'.png')


tif_images(sentinel_rgb, sentinel_rgb, patch_size,
           NAIP_resolution, Sentinel_resolution)
