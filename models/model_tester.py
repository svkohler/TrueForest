import torch
from torch import nn
import sys
from tqdm import tqdm


class Tester(object):
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def test(self, model, data, verbose=True):
        if data == 'train':
            self.dataloader = self.config.train_dataloader
        elif data == 'test':
            self.dataloader = self.config.test_dataloader
        else:
            raise ValueError(
                'please provide data type. either "train" or "test"')
        # load the model
        try:
            checkpoint = torch.load(
                self.config.dump_path + '/'+self.config.model_name+'_best_epoch_' + str(self.config.patch_size)+'.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            if verbose:
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
        elif self.config.model_name == 'SwAV':
            encoder = nn.Sequential(*list(model.module.children())[:-2])
        else:
            encoder = nn.Sequential(
                *list(model.module.encoder.children())[:-1])
        # put model into evaluation mode
        encoder.eval()
        # start testing
        embeddings = torch.tensor([], requires_grad=False).to(self.device)

        if verbose:
            print('Start creating embeddings...')
        for i, (drone, satellite) in enumerate(tqdm(self.dataloader)):
            # send to GPU
            drone = drone.to(self.device)
            satellite = satellite.to(self.device)

            # produce embeddings
            with torch.no_grad():
                drone_emb = encoder(drone)
                sat_emb = encoder(satellite)
            # concat embeddings
            concat = torch.squeeze(torch.cat((drone_emb, sat_emb), dim=1))
            # append embeddings
            embeddings = torch.cat((embeddings, concat), dim=0)
        if verbose:
            print('Successfully created embeddings.')
        return embeddings


class Triplet_tester(object):
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def test(self, data, model):
        if data == 'train':
            self.dataloader = self.config.train_dataloader
        elif data == 'test':
            self.dataloader = self.config.test_dataloader
        else:
            raise ValueError(
                'please provide data type. either "train" or "test"')
        # load the model
        checkpoint = torch.load(
            self.config.dump_path + '/'+self.config.model_name+'_best_epoch_' + str(self.config.patch_size)+'.pth')
        model.module.encoder.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model.')

        # trim the model to the necessary layers
        encoder = nn.Sequential(
            *list(model.module.encoder.children())[:-1])

        # put model into evaluation mode
        encoder.eval()
        # start testing
        embeddings = torch.tensor([], requires_grad=False).to(self.device)

        for i, (drone, satellite) in enumerate(tqdm(self.dataloader)):
            # send to GPU
            drone = drone.to(self.device)
            satellite = satellite.to(self.device)

            # produce embeddings
            with torch.no_grad():
                drone_emb = encoder(drone)
                sat_emb = encoder(satellite)
            # concat embeddings

            concat = torch.squeeze(torch.cat((drone_emb, sat_emb), dim=1))
            # append embeddings
            embeddings = torch.cat((embeddings, concat), dim=0)

        print(embeddings.shape)

        return embeddings
