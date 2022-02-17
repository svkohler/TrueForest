'''
Code adopted with slight changes.
Source: https://github.com/sthalles/PyTorch-BYOL
Date: February 17th, 2022

'''

import torchvision.models as models
import torch
from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class BYOL_Resnet(torch.nn.Module):
    def __init__(self, base_encoder, config):
        super(BYOL_Resnet, self).__init__()

        self.base_encoder = base_encoder(
            pretrained=config.pretrained)

        self.encoder = torch.nn.Sequential(
            *list(self.base_encoder.children())[:-1])

        self.projection = MLPHead(
            in_channels=self.base_encoder.fc.in_features, mlp_hidden_size=config.num_hidden, projection_size=config.num_projection)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)


class BYOL(torch.nn.Module):
    def __init__(self, base_encoder, config):
        super(BYOL, self).__init__()
        self.config = config
        self.online_encoder = BYOL_Resnet(base_encoder, config)
        self.target_encoder = BYOL_Resnet(base_encoder, config)
        self.predictor = MLPHead(
            in_channels=self.online_encoder.projection.net[-1].out_features, mlp_hidden_size=config.num_hidden, projection_size=config.num_features)

    def forward(self, x, y):
        pred_1 = self.predictor(self.online_encoder(x))
        pred_2 = self.predictor(self.online_encoder(y))

        with torch.no_grad():
            target_1 = self.target_encoder(x)
            target_2 = self.target_encoder(y)

        return pred_1, pred_2, target_1, target_2

    @torch.no_grad()
    def _update_target_encoder_params(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.config.ema_factor + \
                param_q.data * (1. - self.config.ema_factor)

    def init_target_encoder(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
