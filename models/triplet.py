'''
Code adopted with slight changes.
Source: https://github.com/KevinMusgrave/pytorch-metric-learning
Date: February 17th, 2022

'''


from pytorch_metric_learning.utils import common_functions
import torch
from torch import nn


class MLP_head(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]
        #self.record_these = ["last_linear", "net"]

    def forward(self, x):
        return self.net(x)


class Triplet(nn.Module):

    def __init__(self, base_encoder, config):
        super(Triplet, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(pretrained=config.pretrained)

        encoder = self.encoder.fc.in_features
        self.encoder.fc = common_functions.Identity()

        self.embedder = MLP_head([encoder, config.num_projection])
