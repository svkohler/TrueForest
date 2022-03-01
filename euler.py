# import system packages
import sys
import os
import socket

# imports for config
from box import Box
import argparse

# import from other files
from utils import *
from models.load_model import *
from models.classifier import *

# information in RAM usage
print('RAM used: ', psutil.virtual_memory()[2])


def get_train_embeddings(config):
    '''
    function to get specifically train embeddings and load them to CPU and convert to numpy array

    '''
    train_embeddings = torch.load(
        config.dump_path+'/embeddings/train_embeddings_' +
        config.model_name+'_Central_Valley_'+str(config.patch_size)+'.pth', map_location='cpu')

    return train_embeddings.cpu().detach().numpy()


def get_test_embeddings(config):
    '''
    function to get specifically test embeddings and load them to CPU and convert to numpy array.
    test embeddings are organised in a dictionary.

    '''
    test_embeddings = {}
    if config.location == 'all':
        for loc in ['Central_Valley', 'Florida', 'Louisiana', 'Tennessee', 'Phoenix']:
            test_embeddings[loc] = torch.load(
                config.dump_path+'/embeddings/test_embeddings_' +
                config.model_name+'_'+loc+'_'+str(config.patch_size)+'.pth', map_location='cpu').cpu().detach().numpy()
    else:
        test_embeddings[config.location] = torch.load(
            config.dump_path+'/embeddings/test_embeddings_' +
            config.model_name+'_'+config.location+'_'+str(config.patch_size)+'.pth', map_location='cpu').cpu().detach().numpy()

    return test_embeddings


# parser to select desired arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    default='custom',
                    # choices=['custom'],
                    help='Select one of the experiments described in our report or setup a custom config file'
                    )
parser.add_argument('--run_mode',
                    default='insert',
                    choices=['train_encoder', 'test_mult', 'train_classifier'],
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
                             'random_forest'],
                    help='type of classifier to use')
parser.add_argument('--num_runs',
                    default=100,
                    type=int,
                    help='how many test runs should be performed')
args = parser.parse_args()

# load config with additional variables
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
print(hostname)
paths_setter(hostname, config)

print('Working on model: ', config.model_name,
      '. With base architecture: ', config.base_architecture)
print('Area Size: ', config.patch_size)
print('Batch Size: ', config.batch_size)
print('RAM used: ', psutil.virtual_memory()[2])

# setting seed for reproduceability
seed_all(config.seed)

# create directory for intermediate objects and results
if not os.path.exists(config.dump_path):
    os.makedirs(config.dump_path)

device = None

# Do for multiple runs: train classifier on training data and subsequently test on test data.
if config.run_mode in ['test_mult']:
    # train_embeddings, test_embeddings = get_embeddings(config)
    train_embeddings = get_train_embeddings(config)
    test_embeddings = get_test_embeddings(config)

    similarity_embeddings(train_embeddings, config)

    test_mult(config, device, train_embeddings,
              test_embeddings, num_runs=config.num_runs)

print('Successful execution.')
