import torch
from torch import nn
import sys


class Tester(object):
    def __init__(self, config, dataloader, device):
        self.config = config
        self.device = device
        self.dataloader = dataloader

    def test(self, model):
        # load the model
        checkpoint = torch.load(
            self.config.dump_path + '/'+self.config.model_name+'_best_epoch.pth')
        model.load_state_dict(checkpoint)
        print('Successfully loaded model.')

        # trim the model to the necessary layers
        encoder = nn.Sequential(*list(model.module.encoder.children())[:-1])

        # put model into evaluation mode
        encoder.eval()
        # start testing
        embeddings = torch.tensor([], requires_grad=False).to(self.device)

        for i, (drone, satellite) in enumerate(self.dataloader):
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

        return embeddings


class SwAV_tester(object):
    def __init__(self, config, dataloader, device):
        self.config = config
        self.device = device
        self.dataloader = dataloader

    def test(self, model):
        # load the model
        checkpoint = torch.load(
            self.config.dump_path + '/'+self.config.model_name+'_best_epoch.pth')
        model.load_state_dict(checkpoint)
        print('Successfully loaded model.')

        # trim the model to the necessary layers
        encoder = nn.Sequential(
            *list(model.module.children())[:-2])

        # put model into evaluation mode
        encoder.eval()
        # start testing
        embeddings = torch.tensor([], requires_grad=False).to(self.device)

        for i, (drone, satellite) in enumerate(self.dataloader):
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
