import torchvision.models as models
import torch
from torch import nn


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, base_encoder, config):
        super().__init__()
        self.config = config
        self.encoder = base_encoder(zero_init_residual=True)
        self.encoder.fc = nn.Identity()

        # projector
        sizes = [2048, self.config.num_hidden, self.config.num_hidden,
                 self.config.num_projection]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.encoder(y1))
        z2 = self.projector(self.encoder(y2))

        return z1, z2
