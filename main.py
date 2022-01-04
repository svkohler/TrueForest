# import system packages
import sys
import os

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
args = parser.parse_args()

# load config
try:
    config = Box.from_yaml(filename="./configs/" + args.config + ".yaml")
except:
    raise OSError("Does not exist", args.config)

config.num_gpus = len(args.gpu_ids)

print('Working on model: ', config.model_name)

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
dataset = TrueForrestDataset(config)

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
    embeddings = tester.test(model)

    print('embeddings shape: ', embeddings.shape)

    classify(config, embeddings.cpu().detach().numpy())

# test binary classifier with test data set
if config.run_mode in ['all', 'test']:
    # replace dataloader by test data
    tester.dataloader = TrueForrestDataset(...)
    # get embeddings from pretrained model
    embeddings = tester.test(model)
    # predict test data
    predict(config, embeddings.cpu().detach().numpy())

print('Successful.')
