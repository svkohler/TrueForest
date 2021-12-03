from torch import nn
import torch
from utils import AverageMeter, ProgressMeter
import time


class SimSiam_trainer():
    def __init__(self, config, dataloader, device, fix_pred_lr=True):
        self.config = config
        self.fix_pred_lr = fix_pred_lr
        self.device = device
        self.dataloader = dataloader

    def train(self, model):
        self.criterion = nn.CosineSimilarity(dim=1)
        if self.fix_pred_lr:
            optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                            {'params': model.predictor.parameters(), 'fix_lr': True}]
        else:
            optim_params = model.parameters()

        self.optimizer = torch.optim.SGD(optim_params, self.config.init_lr,
                                         momentum=self.config.momentum,
                                         weight_decay=self.config.weight_decay)

        for epoch in range(self.config.num_epochs):
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4f')
            progress = ProgressMeter(
                len(self.dataloader),
                [batch_time, data_time, losses],
                prefix="Epoch: [{}]".format(epoch))

            end = time.time()
            for i, (satellite, drone) in enumerate(self.dataloader):
                # send data to GPU
                satellite = satellite.to(self.device)
                drone = drone.to(self.device)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output and loss
                p1, p2, z1, z2 = model(x1=satellite, x2=drone)
                loss = -(self.criterion(p1, z2).mean() +
                         self.criterion(p2, z1).mean()) * 0.5

                losses.update(loss.item(), satellite.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0:
                    progress.display(i)

            print('epoch done')
