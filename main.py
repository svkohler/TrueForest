# import system packages
import sys
import os

# imports for config
from box import Box


def check_venv(venv='mt_env'):
    if sys.prefix.split('/')[-1] != venv:
        raise ConnectionError('Not connected to correct virtual environment')


check_venv()

# dataset = dataloader

# model = load model

# model.train()
