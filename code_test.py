import pickle
import os
from box import Box
import rioxarray as rxr
import earthpy as et

# import rasterio
# from rasterio.plot import show as sh
# from torchvision.transforms.transforms import ToPILImage
import yaml
import torch

# import system packages
import sys
import os
import socket


# imports for config
from box import Box
import argparse

# imports for torch
import torch
from torch import nn

# import from other files
from utils import *
from models.load_model import *
from models.classifier import *

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

import time

import argparse
import math

# from models.classifier import load_clf

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    default='custom',
                    # choices=['custom'],
                    help='Select one of the experiments described in our report or setup a custom config file'
                    )
args = parser.parse_args()

try:
    config = Box.from_yaml(filename="./configs/" + args.config + ".yaml")
except:
    raise OSError("Does not exist", args.config)

# get hostname to set the correct paths
hostname = socket.gethostname()
paths_setter(hostname, config)
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

# end = time.time()

# time.sleep(2)

# print(time.time()-end)

# row = [1, 2, 3, 4]


# def dot_prod(row):
#     return np.dot(row[:int(len(row)/2)], row[int(len(row)/2):])


# def cos_sim(row):
#     return np.dot(row[:int(len(row)/2)], row[int(len(row)/2):])/(np.linalg.norm(row[:int(len(row)/2)])*np.linalg.norm(row[int(len(row)/2):]))


# print(dot_prod(row))
# print(cos_sim(row))
# print(row[:int(len(row)/2)])
# print(row[int(len(row)/2):])

# def produce_negative_samples(data):

#     data_copy = data.copy()
#     data_copy = data_copy[1:, :]
#     data_copy = np.append(data_copy, [data[0, :]], axis=0)

#     return np.concatenate((data[:, :int(data.shape[1]/2)], data_copy[:, int(data_copy.shape[1]/2):]), axis=1)


# data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# print(produce_negative_samples(data))

# def adjust_learning_rate(epoch):
#     """Decays the learning rate with half-cycle cosine after warmup"""
#     if epoch < config.warm_up_epochs:
#         lr = config.init_lr * epoch / config.warm_up_epochs
#     else:
#         lr = config.init_lr * 0.5 * (1. + math.cos(math.pi * (
#             epoch - config.warm_up_epochs) / (config.num_epochs - config.warm_up_epochs)))
#     return lr


# for epoch in range(100):
#     print(adjust_learning_rate(epoch))


# data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


# def dot_sim(row):
#     return np.dot(row[:int(len(row)/2)], row[int(len(row)/2):])


# def cos_sim(row):
#     return np.dot(row[:int(len(row)/2)], row[int(len(row)/2):])/(np.linalg.norm(row[:int(len(row)/2)])*np.linalg.norm(row[int(len(row)/2):]))


# def mse(row):
#     return np.square(np.subtract(row[:int(len(row)/2)], row[int(len(row)/2):])).mean()


# pos_dot = np.apply_along_axis(dot_sim, 1, data)
# pos_cos = np.apply_along_axis(cos_sim, 1, data)
# pos_mse = np.apply_along_axis(mse, 1, data)

# print(pos_dot)

# def trim_stat(arr, upper_quantile=0.99, lower_quantile=0.01, stat='mean'):
#     upper_q = np.quantile(arr, upper_quantile)
#     lower_q = np.quantile(arr, lower_quantile)

#     arr_new = [x for x in arr if upper_q > x > lower_q]

#     if stat == 'mean':
#         return np.array(arr_new).std()

#     if stat == 'std':
#         return np.array(arr_new).std()

#     print('Error: stat not implemented.')


# arr = np.array([1, 2, 3, 4, 5, 6, 10, 10, 10])

with open(config.dump_path + '/accuracies/'+config.model_name+'_'+str(config.patch_size)+'_test_accuracies_'+config.clf+'.pkl', 'rb') as data:
    acc = pickle.load(data)

print(acc.dict)
# print(len(acc.dict['Central_Valley']))

# d = dict((i, value)
#          for (i, (key, value)) in enumerate(acc.dict['Central_Valley'].items()))
# print(d)

# a = np.array([1, 2, 3, 0, 0, 0])

# print(sum(a == 0))

# for i in range(7, 10):
#     print(i)


# acc = np.array([0.3, 0.4, 0.5, 0, 0.6, 1])

# acc = np.delete(acc, np.where(acc == 0))
# acc = np.delete(acc, np.where(acc == 1))

# arr = np.zeros(10)

# arr[:len(acc)] = acc


# print(arr)

# def get_embeddings(config):
#     train_embeddings = torch.load(
#         config.dump_path+'/train_embeddings_' +
#         config.model_name+'_'+str(config.patch_size)+'.pth')
#     test_embeddings = torch.load(
#         config.dump_path+'/test_embeddings_' +
#         config.model_name+'_'+str(config.patch_size)+'.pth')

#     return train_embeddings.cpu().detach().numpy(), test_embeddings.cpu().detach().numpy()


# emb = get_embeddings(config)

# # for i in range(5):

# #     neg = produce_negative_samples(emb[0])

# #     print(neg)

# print(emb)

# print(np.arange(emb[0].shape[0]))

# r = np.arange(emb[0].shape[0])

# np.random.shuffle(r)

# print(r)


# arr = np.array([[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4], [
#                5, 5, 5, 6, 6, 6], [7, 7, 7, 8, 8, 8], [9, 9, 9, 10, 10, 10]])

# np.random.shuffle(arr)

# print(produce_negative_samples(arr))

# drone_imgs = os.listdir(config.data_store + '/drone/Central_Valley/train/224')
# sat_imgs = os.listdir(config.data_store +
#                       '/satellite_rgb/Central_Valley/train/224')

# drone_num = []
# sat_num = []
# for i in drone_imgs:
#     drone_num.append(i[6:])

# for i in sat_imgs:
#     sat_num.append(i[10:])

# for i in sat_num:
#     if i not in drone_num:
#         print(i)

# # print(drone_num)
# # print(sat_num)

# d = {}

# d[1] = (1, 2)
# d[2] = (3, 4)

# d1 = dict(zip([3, 4], list(d.values())))

# print(d)
# print(d1)

# acc = AccuracyCollector(num_runs=100)
# while acc.runs < 100:
#     for loc in ['a', 'b', 'c']:
#         res = np.random.rand(5)
#         acc.update(loc, tuple(res), acc.runs)
#     acc.update_runs()

# acc.end_statement()

# res = torch.load(
#     '/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/dump_from_remote/SimSiam_best_epoch_224.pth', map_location='cuda:0')

# print(res['epoch'])
