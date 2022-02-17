# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Code adopted with slight changes.
Source: https://github.com/facebookresearch/simsiam
Date: February 17th, 2022

'''

import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(pretrained=config.pretrained,
                                    zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]

        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # first layer
                                        nn.Linear(
                                            prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # second layer
                                        nn.Linear(
                                            in_features=prev_dim, out_features=config.num_features, bias=True),
                                        nn.BatchNorm1d(config.num_projection, affine=False))  # output layer
        # hack: not use bias as it is followed by BN
        self.encoder.fc[6].bias.requires_grad = False

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(config.num_projection, config.num_hidden, bias=False),
                                       nn.BatchNorm1d(config.num_hidden),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(config.num_hidden, config.num_features))  # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        # here when evaluating z1, z2 are used
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        # print("\tIn Model: input size", x1.size(),
        #       "output size", z1.size())

        return p1, p2, z1.detach(), z2.detach()
