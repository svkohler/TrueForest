'''
Code adopted with slight changes.
Source: https://github.com/facebookresearch/moco
Date: February 17th, 2022

'''


import torch
from torch import nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, config, dim=128, K=65536):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.config = config

        self.K = config.queue_length
        self.m = config.ema_factor
        self.T = config.temperature

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(pretrained=config.pretrained)
        self.encoder_k = base_encoder(pretrained=config.pretrained)

        self._build_projector_and_predictor_mlps_ResNet()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(
            self.config.num_projection, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _build_mlp(self, num_layers, input_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else self.config.num_hidden
            dim2 = self.config.num_projection if l == num_layers - \
                1 else self.config.num_hidden

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps_ResNet(self):
        hidden_dim = self.encoder_q.fc.weight.shape[1]
        del self.encoder_q.fc, self.encoder_k.fc  # remove original fc layer

        # projectors
        self.encoder_q.fc = self._build_mlp(
            2, hidden_dim)
        self.encoder_k.fc = self._build_mlp(
            2, hidden_dim)

        # predictor
        self.predictor = self._build_mlp(
            2, self.config.num_projection, False)
        print(self.predictor)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, drone, satellite):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        drone_q = self.predictor(self.encoder_q(drone))  # queries: NxC
        drone_q = nn.functional.normalize(drone_q, dim=1)
        sat_q = self.predictor(self.encoder_q(satellite))  # queries: NxC
        sat_q = nn.functional.normalize(sat_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            drone_k = self.encoder_k(drone)  # keys: NxC
            drone_k = nn.functional.normalize(drone_k, dim=1)
            sat_k = self.encoder_k(satellite)  # keys: NxC
            sat_k = nn.functional.normalize(sat_k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        drone_l_pos = torch.einsum('nc,nc->n', [drone_q, sat_k]).unsqueeze(-1)
        sat_l_pos = torch.einsum('nc,nc->n', [sat_q, drone_k]).unsqueeze(-1)
        # negative logits: NxK
        drone_l_neg = torch.einsum(
            'nc,ck->nk', [drone_q, self.queue.clone().detach()])
        sat_l_neg = torch.einsum(
            'nc,ck->nk', [sat_q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        drone_logits = torch.cat([drone_l_pos, drone_l_neg], dim=1)
        sat_logits = torch.cat([sat_l_pos, sat_l_neg], dim=1)

        # apply temperature
        drone_logits /= self.T
        sat_logits /= self.T

        # labels: positive key indicators
        drone_labels = torch.zeros(
            drone_logits.shape[0], dtype=torch.long).cuda()
        sat_labels = torch.zeros(sat_logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(torch.cat((drone_k, sat_k), 0))

        return drone_logits, drone_labels, sat_logits, sat_labels
