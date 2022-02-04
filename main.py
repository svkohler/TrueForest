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

print('RAM used: ', psutil.virtual_memory()[2])


# parser to select desired arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    default='custom',
                    # choices=['custom'],
                    help='Select one of the experiments described in our report or setup a custom config file'
                    )
parser.add_argument('--run_mode',
                    default='insert',
                    choices=['train_encoder', 'test_mult'],
                    help='Select run mode.'
                    )
parser.add_argument('--gpu_ids',
                    default=[0],
                    nargs="+",
                    type=int,
                    help='select IDs of GPUs to use')
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
                    help='how frequently info is printed')
parser.add_argument('--clf',
                    default='insert',
                    type=str,
                    choices=['linear', 'MLP', 'xgboost',
                             'random_forest', 'MLP_1'],
                    help='type of classifier to use')
parser.add_argument('--num_runs',
                    default=100,
                    type=int,
                    help='how many test runs should be performed')
args = parser.parse_args()

# load config
try:
    config = Box.from_yaml(filename="./configs/" + args.config + ".yaml")
except:
    raise OSError("Does not exist", args.config)

# process command line input variables
config.num_gpus = len(args.gpu_ids)
config.num_runs = args.num_runs
config.gpu_ids = args.gpu_ids

if args.batch_size != -1:
    config.batch_size = args.batch_size

if args.patch_size != -1:
    config.patch_size = args.patch_size

if args.print_freq != -1:
    config.print_freq = args.print_freq

if args.run_mode != 'insert':
    config.run_mode = args.run_mode

if args.clf != 'insert':
    config.clf = args.clf

# get hostname to set the correct paths
hostname = socket.gethostname()
paths_setter(hostname, config)

print('Working on model: ', config.model_name,
      '. With base architecture: ', config.base_architecture)
print('Area Size: ', config.patch_size)
print('Batch Size: ', config.batch_size)
print('RAM used: ', psutil.virtual_memory()[2])

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

# create training dataset
train_dataset = TrueForestDataset(config, mode='train')
test_dataset = TrueForestDataset(config, mode='test', transform=False)

# create the dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                                               num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                                              num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)

# train_dataloader, test_dataloader = create_dataloader(config)

config.train_dataloader = train_dataloader
config.test_dataloader = test_dataloader

# load the model
model, trainer, tester = load_model(config, device)

# initiate parallel GPUs
print("Your setup has ", torch.cuda.device_count(), "GPUs.")
# wrap model for multiple GPU usage
model = nn.DataParallel(model, args.gpu_ids)
# send model to GPU
model.to(device)
print('RAM used: ', psutil.virtual_memory()[2])

# train the model and save best version
if config.run_mode in ['train_encoder']:
    trainer.train(model)

# Do for multiple runs: train classifier on training data and subsequently test on test data.
if config.run_mode in ['test_mult']:
    create_embeddings(config, model, tester)
    train_embeddings, test_embeddings = get_embeddings(config)
    # train_embeddings = get_train_embeddings(config)
    # test_embeddings = get_test_embeddings(config)

    similarity_embeddings(train_embeddings, config)

    test_mult(config, device, train_embeddings,
              test_embeddings, num_runs=config.num_runs)

print('Successful execution.')
