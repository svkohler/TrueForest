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


# parser to select desired
parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    default='custom',
                    choices=['custom'],
                    help='Select one of the experiments described in our report or setup a custom config file'
                    )
args = parser.parse_args()

# load config
try:
    config = Box.from_yaml(filename="./configs/" + args.config + ".yaml")
except:
    raise OSError("Does not exist", args.config)

# check if connected to virtual environment
check_venv()

#
if not os.path.exists(config.dump_path):
    os.makedirs(config.dump_path)

# define the device where computations are run
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create the dataset
dataset = TrueForrestDataset(config)

# create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                                         num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)

# load the model
model, trainer = load_model(config, dataloader, device)

# initiate parallel GPUs
print("You have ", torch.cuda.device_count(), "GPUs available.")

# wrap model for multiple GPU usage
model = nn.DataParallel(model)

# send model to GPU
model.to(device)

# train the model
trainer.train(model)

print('Successful.')
