from torch import nn
import torch
from utils import AverageMeter, ProgressMeter, accuracy
import time
import os
import sys
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from utils import LARC
import numpy as np
import math
import torch.distributed as dist
import apex


# ------------------- SimSiam trainer -------------------- #

class SimSiam_trainer(object):
    def __init__(self, config, dataloader, device, fix_pred_lr=True):
        self.config = config
        self.fix_pred_lr = fix_pred_lr
        self.device = device
        self.dataloader = dataloader

    def train(self, model):
        # define the loss criterion
        self.criterion = nn.CosineSimilarity(dim=1)
        # fix learning rate of the predictor
        if self.fix_pred_lr:
            optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                            {'params': model.predictor.parameters(), 'fix_lr': True}]
        else:
            optim_params = model.parameters()

        # define the optimizer
        self.optimizer = torch.optim.SGD(optim_params, self.config.init_lr,
                                         momentum=self.config.momentum,
                                         weight_decay=self.config.weight_decay)

        # define the scaler for mixed precision
        scaler = GradScaler(enabled=self.config.fp16_precision)

        # check for optimal backend
        torch.backends.cudnn.benchmark = True

        # put model in training mode
        model.train()

        #  *** main training loop ***
        for epoch in range(self.config.num_epochs):
            # initialize meters to keep track of stats
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4f')
            progress = ProgressMeter(
                len(self.dataloader),
                [batch_time, data_time, losses],
                prefix="Epoch: [{}]".format(epoch))

            end = time.time()

            # loop through batches
            for i, (satellite, drone) in enumerate(self.dataloader):
                # send data to GPU
                satellite = satellite.to(self.device)
                drone = drone.to(self.device)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output and loss
                with autocast(enabled=self.config.fp16_precision):
                    p1, p2, z1, z2 = model(x1=satellite, x2=drone)
                    loss = -(self.criterion(p1, z2).mean() +
                             self.criterion(p2, z1).mean()) * 0.5

                losses.update(loss.item(), satellite.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0:
                    progress.display(i)


# ------------------- SimCLR trainer -------------------- #

class SimCLR_trainer(object):

    def __init__(self, config, dataloader, device):
        self.config = config
        self.dataloader = dataloader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def info_nce_loss(self, features_d, features_sat):

        features_d = F.normalize(features_d, dim=1)
        features_sat = F.normalize(features_sat, dim=1)

        similarity_matrix = torch.matmul(features_d, features_sat.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(
            self.device)

        # select and combine multiple positives
        positives = similarity_matrix[mask.bool()].view(
            similarity_matrix.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~mask.bool()].view(
            similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(
            logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.config.temperature
        return logits, labels

    def train(self, model):

        # define the optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), self.config.init_lr, weight_decay=self.config.weight_decay)

        # define learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.dataloader), eta_min=0,
                                                                    last_epoch=-1)
        # define scaler for mixed precision
        scaler = GradScaler(enabled=self.config.fp16_precision)

        # check for optimal backend
        torch.backends.cudnn.benchmark = True

        # put model in training mode
        model.train()

        #  *** main training loop ***
        for epoch in range(self.config.num_epochs):
            # initialize meters to keep track of stats
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4f')
            acc = AverageMeter('Accuracy', ':.4f')
            progress = ProgressMeter(
                len(self.dataloader),
                [batch_time, data_time, losses, acc],
                prefix="Epoch: [{}]".format(epoch))

            end = time.time()

            # loop through batches
            for i, (satellite, drone) in enumerate(self.dataloader):
                # send data to GPU
                drone = drone.to(self.device)
                satellite = satellite.to(self.device)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output and loss
                with autocast(enabled=self.config.fp16_precision):
                    features_d = model(drone)
                    features_sat = model(satellite)
                    logits, labels = self.info_nce_loss(
                        features_d, features_sat)
                    loss = self.criterion(logits, labels)

                losses.update(loss.item(), satellite.size(0))

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    acc.update(top1.cpu().numpy()[0])
                    progress.display(i)

            # warmup for the first 10 epochs
            if epoch >= 10:
                self.scheduler.step()


# ------------------- BYOL trainer -------------------- #

class BYOL_trainer(object):

    def __init__(self, config, dataloader, device):
        self.config = config
        self.dataloader = dataloader
        self.device = device

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def train(self, model):

        # define the optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), self.config.init_lr, weight_decay=self.config.weight_decay)

        # define scaler for mixed precision
        scaler = GradScaler(enabled=self.config.fp16_precision)

        # check for optimal backend
        torch.backends.cudnn.benchmark = True

        # put model in training mode
        model.train()

        #  *** main training loop ***
        for epoch in range(self.config.num_epochs):
            # initialize meters to keep track of stats
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4f')
            acc = AverageMeter('Accuracy', ':.4f')
            progress = ProgressMeter(
                len(self.dataloader),
                [batch_time, data_time, losses, acc],
                prefix="Epoch: [{}]".format(epoch))

            end = time.time()

            # loop through batches
            for i, (satellite, drone) in enumerate(self.dataloader):
                # send data to GPU
                drone = drone.to(self.device)
                satellite = satellite.to(self.device)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output and loss
                with autocast(enabled=self.config.fp16_precision):
                    online_pred_drone, target_proj_drone = model(drone)
                    online_pred_satellite, target_proj_satellite = model(
                        satellite)
                    loss_one = self.loss_fn(
                        online_pred_drone, target_proj_satellite.detach())
                    loss_two = self.loss_fn(
                        online_pred_satellite, target_proj_drone.detach())

                    loss = (loss_one + loss_two).mean()

                losses.update(loss.item(), satellite.size(0))

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                model.module.update_moving_average()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            # warmup for the first 10 epochs
            if epoch >= 10:
                self.scheduler.step()


# ------------------- SwAV trainer -------------------- #


class SwAV_trainer(object):

    def __init__(self, config, dataloader, device):
        self.config = config
        self.dataloader = dataloader
        self.device = device

    def train(self, model):

        # define optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.init_lr,
            momentum=0.9,
            weight_decay=self.config.weight_decay,
        )

        # wrap optimzer in LARC module
        self.optimizer = LARC(optimizer=optimizer,
                              trust_coefficient=0.001, clip=False)

        # define the learning rate scheduler
        warmup_lr_schedule = np.linspace(
            self.config.init_warm_up_lr, self.config.init_lr, len(self.dataloader) * self.config.warm_up_epochs)
        iters = np.arange(len(self.dataloader) *
                          (self.config.num_epochs - self.config.warm_up_epochs))
        cosine_lr_schedule = np.array([self.config.final_lr + 0.5 * (self.config.init_lr - self.config.final_lr) * (1 +
                                                                                                                    math.cos(math.pi * t / (len(self.dataloader) * (self.config.num_epochs - self.config.warm_up_epochs)))) for t in iters])
        self.scheduler = np.concatenate(
            (warmup_lr_schedule, cosine_lr_schedule))

        # build the queue to increase number of examples used for cluster assignment
        queue = None
        queue_path = os.path.join(self.config.dump_path, "SwAV_queue.pth")
        if os.path.isfile(queue_path):
            queue = torch.load(queue_path)["queue"]
        # the queue needs to be divisible by the batch size
        self.config.queue_length -= self.config.queue_length % (
            self.config.batch_size)

        # check for optimal backend
        torch.backends.cudnn.benchmark = True

        # put model into training mode
        model.train()

        #  *** main training loop ***
        for epoch in range(self.config.num_epochs):
            # initialize meters to keep track of stats
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4f')
            progress = ProgressMeter(
                len(self.dataloader),
                [batch_time, data_time, losses],
                prefix="Epoch: [{}]".format(epoch))

            # optionally starts a queue
            if self.config.queue_length > 0 and epoch >= self.config.epoch_queue_starts and queue is None:
                queue = self.init_queue()

            # set flag
            use_the_queue = False

            end = time.time()

            # loop through batches
            for i, (satellite, drone) in enumerate(self.dataloader):
                # send data to GPU
                drone = drone.to(self.device)
                satellite = satellite.to(self.device)

                # measure data loading time
                data_time.update(time.time() - end)

                # update learning rate
                iteration = epoch * len(self.dataloader) + i
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.scheduler[iteration]

                # normalize the prototypes
                with torch.no_grad():
                    w = model.module.prototypes.weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    model.module.prototypes.weight.copy_(w)

                # compute embeddings and cluster assignments
                embedding_drone, output_drone = model(drone)
                embedding_sat, output_sat = model(satellite)

                # combine model outputs
                embedding = torch.zeros(
                    self.config.batch_size*2, self.config.num_features)
                output = torch.zeros(
                    self.config.batch_size*2, self.config.num_prototypes)
                embedding[::2, :] = embedding_drone
                embedding[1::2, :] = embedding_sat
                output[::2, :] = output_drone
                output[1::2, :] = output_sat

                embedding = embedding.detach()
                output = output.detach()

                # save batch size
                bs = self.config.batch_size

                # compute the SwAV loss
                loss = torch.tensor(
                    0, dtype=torch.float64, requires_grad=True).to(self.device)
                for j, crop_id in enumerate([0, 1]):
                    with torch.no_grad():
                        out = output[bs * crop_id: bs *
                                     (crop_id + 1)].detach().to(self.device)
                        # time to use the queue
                        if queue is not None:
                            if use_the_queue or not torch.all(queue[j, -1, :] == 0):
                                use_the_queue = True
                                out = torch.cat((torch.mm(
                                    queue[j],
                                    model.module.prototypes.weight.t()
                                ), out))
                            # fill the queue
                            queue[j, bs:] = queue[j, :-bs].clone()
                            queue[j, :bs] = embedding[crop_id *
                                                      bs: (crop_id + 1) * bs]

                        # get assignments
                        q = self.distributed_sinkhorn(
                            out)[-bs:].to(self.device)

                    # cluster assignment prediction
                    subloss = torch.tensor(
                        0, dtype=torch.float64, requires_grad=True).to(self.device)
                    for v in np.delete(np.arange(2), crop_id):
                        x = output[bs * v: bs *
                                   (v + 1)].to(self.device) / self.config.temperature
                        subloss -= torch.mean(torch.sum(q *
                                                        F.log_softmax(x, dim=1), dim=1))
                    loss += subloss / (np.sum(2) - 1)
                loss /= len([0, 1])

                losses.update(loss.item(), self.config.batch_size)

                self.optimizer.zero_grad()

                loss.backward()

                # cancel gradients for the prototypes at the beginning
                if iteration < self.config.freeze_prototypes_niters:
                    for name, p in model.named_parameters():
                        if "prototypes" in name:
                            p.grad = None

                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0:
                    progress.display(i)

            if queue is not None:
                torch.save({"queue": queue}, queue_path)

    def init_queue(self):
        return torch.zeros(
            len([0, 1]),
            self.config.queue_length // self.config.num_gpus,
            self.config.num_features,
        ).cuda()

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        # Q is K-by-B for consistency with notations from our paper
        Q = torch.exp(out / self.config.epsilon).t()
        B = Q.shape[1] * self.config.num_gpus  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.config.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            # dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
