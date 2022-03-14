import torch
from torch import nn
import sys
from tqdm import tqdm

'''
This file contains the tester objects.

'''


# ------------------- Basic Tester -------------------- #


class Tester(object):
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def test(self, model, data, location='Central_Valley'):
        if data == 'train':
            self.dataloader = self.config.train_dataloader
        elif data == 'test':
            self.dataloader = self.config.test_dataloader[location]
        else:
            raise ValueError(
                'please provide data type. either "train" or "test"')
        # load the model
        try:
            checkpoint = torch.load(
                self.config.dump_path + '/'+self.config.model_name+'_best_epoch_' + str(self.config.patch_size)+'.pth', map_location=f'cuda:{self.config.gpu_ids[0]}')
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Successfully loaded model.')
        except ValueError:
            print('there seems to be no correspondig model state saved in ',
                  self.config.dump_path)

        # trim the model to the necessary layers
        if self.config.model_name == 'BYOL':
            encoder = nn.Sequential(
                *list(model.module.online_encoder.encoder.children()))
        elif self.config.model_name == 'MoCo':
            encoder = nn.Sequential(
                *list(model.module.encoder_q.children())[:-1])
        else:
            encoder = nn.Sequential(
                *list(model.module.encoder.children())[:-1])
        # put model into evaluation mode
        encoder.eval()
        # start testing
        embeddings = torch.tensor([], requires_grad=False).to(self.device)
        import matplotlib.pyplot as plt
        print('Start creating embeddings...')
        for i, (satellite, drone) in enumerate(tqdm(self.dataloader)):
            # send to GPU
            satellite = satellite.to(self.device)
            drone = drone.to(self.device)

            # produce embeddings
            with torch.no_grad():
                sat_emb = encoder(satellite)
                drone_emb = encoder(drone)
            # concat embeddings
            concat = torch.squeeze(torch.cat((sat_emb, drone_emb), dim=1))
            # append embeddings
            embeddings = torch.cat((embeddings, concat), dim=0)
        print('Successfully created embeddings.')
        return embeddings


# ------------------- Triplet Tester -------------------- #


class MetricLearning_tester(object):
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def test(self, model, data, location='Central_Valley'):
        if data == 'train':
            self.dataloader = self.config.train_dataloader
        elif data == 'test':
            self.dataloader = self.config.test_dataloader[location]
        else:
            raise ValueError(
                'please provide data type. either "train" or "test"')
        # load the model
        checkpoint = torch.load(
            self.config.dump_path + '/'+self.config.model_name+'_best_epoch_' + str(self.config.patch_size)+'.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model.')

        # trim the model to the necessary layers
        encoder = nn.Sequential(
            *list(model.module.encoder.children())[:-1])

        # put model into evaluation mode
        encoder.eval()
        # start testing
        embeddings = torch.tensor([], requires_grad=False).to(self.device)
        for i, (satellite, drone) in enumerate(tqdm(self.dataloader)):

            # send to GPU
            satellite = satellite.to(self.device)
            drone = drone.to(self.device)

            # produce embeddings
            with torch.no_grad():
                sat_emb = encoder(satellite)
                drone_emb = encoder(drone)
            # concat embeddings

            concat = torch.squeeze(torch.cat((sat_emb, drone_emb), dim=1))
            # append embeddings
            embeddings = torch.cat((embeddings, concat), dim=0)

        return embeddings
