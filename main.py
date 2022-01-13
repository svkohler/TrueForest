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

# parser to select desired
parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    default='custom',
                    # choices=['custom'],
                    help='Select one of the experiments described in our report or setup a custom config file'
                    )
parser.add_argument('--gpu_ids',
                    default=[0],
                    nargs="+",
                    type=int,
                    help='select IDs of GPUs to use,')
parser.add_argument('--batch_size',
                    default=-1,
                    type=int,
                    help='insert batch size to overwrite config file')
parser.add_argument('--patch_size',
                    default=-1,
                    type=int,
                    help='insert area size')
parser.add_argument('--print_freq',
                    default=-1,
                    type=int,
                    help='insert area size')
args = parser.parse_args()

# load config
try:
    config = Box.from_yaml(filename="./configs/" + args.config + ".yaml")
except:
    raise OSError("Does not exist", args.config)

# process command line input variables
config.num_gpus = len(args.gpu_ids)

if args.batch_size != -1:
    config.batch_size = args.batch_size

if args.patch_size != -1:
    config.patch_size = args.patch_size

if args.print_freq != -1:
    config.print_freq = args.print_freq


hostname = socket.gethostname()
paths_setter(hostname, config)

print('Working on model: ', config.model_name,
      '. With base architecture: ', config.base_architecture)
print('Area Size: ', config.patch_size)
print('Batch Size: ', config.batch_size)

# check if connected to virtual environment
# check_venv()

# setting seed for reproduceability
seed_all(config.seed)

# create directory for intermediate objects and results
if not os.path.exists(config.dump_path):
    os.makedirs(config.dump_path)

# define the device where computations are run
device = torch.device(
    f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

# create the dataset
dataset = TrueForrestDataset(config, mode='train')

# create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                                         num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)

# load the model
model, trainer, tester = load_model(config, dataloader, device)
# initiate parallel GPUs
print("You have ", torch.cuda.device_count(), "GPUs available.")

# wrap model for multiple GPU usage
model = nn.DataParallel(model, args.gpu_ids)

# send model to GPU
model.to(device)

# train the model and save best version
if config.run_mode in ['all', 'train', 'train_encoder']:
    trainer.train(model)

# get embeddings from trained model and train a binary classifier with train dataset
if config.run_mode in ['all', 'train', 'train_classifier']:
    if os.path.isfile(config.dump_path+'/embeddings_'+config.model_name+'_'+str(config.patch_size)+'.pth') == False:
        embeddings = tester.test(model)
        torch.save(embeddings, config.dump_path+'/embeddings_' +
                   config.model_name+'_'+str(config.patch_size)+'.pth')
    else:
        embeddings = torch.load(
            config.dump_path+'/embeddings_' +
            config.model_name+'_'+str(config.patch_size)+'.pth')

    print('embeddings shape: ', embeddings.shape)

    # get embeddings on CPU
    embeddings = embeddings.cpu().detach().numpy()

    # calculate similarity measures
    print('Similarities of embeddings: ')
    similarities = similarity_embeddings(embeddings, config)
    print(similarities)

    classify(config, embeddings)

# test binary classifier with test data set
if config.run_mode in ['test']:
    # replace dataloader by test data
    dataset = TrueForrestDataset(config, mode='test')
    tester.dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                                                    num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)
    # get embeddings from pretrained model
    embeddings = tester.test(model)
    # predict test data
    predict(config, embeddings.cpu().detach().numpy())

print('Successful.')
