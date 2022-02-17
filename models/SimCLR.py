'''
Code adopted with slight changes.
Source: https://github.com/sthalles/SimCLR
Date: February 17th, 2022

'''

import torch.nn as nn


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, config):
        super(ResNetSimCLR, self).__init__()

        self.encoder = base_model(
            pretrained=config.pretrained)
        dim_mlp = self.encoder.fc.in_features
        # add mlp projection head
        self.encoder.fc = nn.Sequential(
            nn.Linear(dim_mlp, config.num_hidden), nn.ReLU(), nn.Linear(in_features=dim_mlp, out_features=config.num_features, bias=True))

    def forward(self, x):
        return self.encoder(x)
