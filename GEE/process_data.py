# Imports
import socket
from tif_to_png import tif2png
from tqdm import tqdm
import argparse
import yaml
from skimage import io
import scipy.ndimage
import scipy
from box import Box
import numpy as np
from PIL import Image
import os
import sys


def paths_setter(hostname, config):
    if hostname == 'svkohler':
        config.data_store = "/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data"
        config.dump_path = "/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/dump"

    if hostname == 'spaceml1.ethz.ch':
        config.data_store = "/mnt/ds3lab-scratch/svkohler/data"
        config.dump_path = "/mnt/ds3lab-scratch/svkohler/dump"


parser = argparse.ArgumentParser()
parser.add_argument('--patch_size',
                    type=int,
                    help='Choose patch size for images in meters',
                    choices=[224, 448, 672, 896, 1120]
                    )
parser.add_argument('--data_type',
                    type=str,
                    help='Whether it is train or test data',
                    choices=['train', 'test']
                    )
parser.add_argument('--location',
                    type=str,
                    help='Location of images',
                    choices=['Central_Valley', 'Florida',
                             'Tennessee', 'Louisiana', 'Phoenix']
                    )
args = parser.parse_args()

# resolution in m/px
NAIP_RESOLUTION = 1
SENTINEL_RESOLUTION = 10

# output image size for input in model
IMAGE_SIZE: 224

# load config
try:
    config = Box.from_yaml(
        filename="/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/configs/custom.yaml")
except:
    raise OSError("Does not exist")

hostname = socket.gethostname()
paths_setter(hostname, config)

# patch size in meters
patch_size = int(args.patch_size)

# load images
Image.MAX_IMAGE_PIXELS = 9999999999

# define paths
if config.model_name == 'Triplet':
    paths = {
        'raw_naip': config.data_store + '/raw/NAIP/'+args.location+'/'+args.data_type,
        'raw_sen_rgb': config.data_store + '/raw/Sentinel_RGB/'+args.location+'/'+args.data_type,
        'drone': config.data_store + '/triplet/'+args.location+'/'+args.data_type+'/' + str(patch_size),
        'sat_rgb': config.data_store+'/triplet/'+args.location+'/'+args.data_type+'/' + str(patch_size),
    }
else:
    paths = {
        'raw_naip': config.data_store + '/raw/NAIP/'+args.location+'/'+args.data_type,
        'raw_sen_rgb': config.data_store + '/raw/Sentinel_RGB/'+args.location+'/'+args.data_type,
        'drone': config.data_store + '/drone/'+args.location+'/'+args.data_type+'/' + str(patch_size),
        'sat_rgb': config.data_store+'/satellite_rgb/'+args.location+'/'+args.data_type+'/' + str(patch_size),
    }

# get the list of naip, sentinel_RGB, sentinel_NIR image patches
naip_patches = os.listdir(paths['raw_naip'])
sentinel_RGB_patches = os.listdir(paths['raw_sen_rgb'])
# extract coordinates from image path to later match against
sentinel_RGB_coordinates = [
    patch[19:len(patch)-4] for patch in sentinel_RGB_patches]

# loop over naip image patches and search for the corresponding sentinel satellite patch
converter = tif2png()
for patch in naip_patches:
    # get naip coordinated
    coordinates = patch[11:len(patch)-4]
    # find the indices of the corresponding sentinel images
    idx_RGB = sentinel_RGB_coordinates.index(coordinates)

    # read the images
    naip = Image.fromarray(
        np.uint8(io.imread(paths['raw_naip'] + '/'+patch))[:, :, :3])
    sentinel_rgb = Image.fromarray(
        np.uint8(io.imread(paths['raw_sen_rgb'] + '/'+sentinel_RGB_patches[idx_RGB])))

    print('Loaded coordiantes: ', coordinates)

    converter(config, paths, naip, sentinel_rgb, patch_size,
              NAIP_RESOLUTION, SENTINEL_RESOLUTION)

    print('Succesfully cropped coordiantes: ', coordinates)
