import os
import sys
from torchvision import transforms, datasets
from pytorch_metric_learning import losses, miners, samplers, trainers
from torch import nn
import torch
from utils import AverageMeter, ProgressMeter
import time
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from utils import LARS
import math

'''
This file contains all the different trainer object for the various SSL approaches

'''


# ------------------- SimSiam trainer -------------------- #

'''
Code adopted with slight changes.
Source: https://github.com/facebookresearch/simsiam
Date: February 17th, 2022

'''


class SimSiam_trainer(object):
    def __init__(self, config, device, fix_pred_lr=True):
        self.config = config
        self.fix_pred_lr = fix_pred_lr
        self.device = device
        self.dataloader = config.train_dataloader

    def adjust_learning_rate(self, optimizer, epoch, config):
        """Decay the learning rate based on schedule"""
        cur_lr = config.init_lr * 0.5 * \
            (1. + math.cos(math.pi * epoch / config.num_epochs))
        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = config.init_lr
            else:
                param_group['lr'] = cur_lr

    def train(self, model):
        # define the loss criterion
        self.criterion = nn.CosineSimilarity(dim=1)
        # fix learning rate of the predictor
        if self.fix_pred_lr:
            optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                            {'params': model.module.predictor.parameters(), 'fix_lr': True}]
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
        losses = AverageMeter('Loss', ':.4f')
        for epoch in range(self.config.num_epochs):
            # initialize meters to keep track of stats
            batch_time = AverageMeter('Data processing time (avg)', ':6.3f')
            data_time = AverageMeter('Data loading time (avg)', ':6.3f')
            progress = ProgressMeter(
                len(self.dataloader),
                [batch_time, data_time, losses],
                prefix="Epoch: [{}]".format(epoch))

            end = time.time()

            self.adjust_learning_rate(self.optimizer, epoch, self.config)

            # loop through batches
            for i, (satellite, drone) in enumerate(self.dataloader):
                # send data to GPU

                satellite = satellite.to(self.device)
                drone = drone.to(self.device)

                # measure data loading time
                data_time.update(time.time() - end)
                end = time.time()

                # compute output and loss
                with autocast(enabled=self.config.fp16_precision):
                    p1, p2, z1, z2 = model(x1=satellite, x2=drone)
                    loss = -(self.criterion(p1, z2).mean() +
                             self.criterion(p2, z1).mean()) * 0.5

                losses.update(loss.item(), self.config.batch_size)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0 or i == len(self.dataloader)-1:
                    progress.display(i)

            # check if current epoch is best epoch and save model state_dict
            losses.check_best_epoch(model, epoch, self.config)


# ------------------- SimCLR trainer -------------------- #

'''
Code adopted with slight changes.
Source: https://github.com/sthalles/SimCLR
Date: February 17th, 2022

'''


class SimCLR_trainer(object):

    def __init__(self, config, device):
        self.config = config
        self.dataloader = config.train_dataloader
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
        self.optimizer = LARS(model.parameters(), lr=self.config.init_lr, weight_decay=self.config.weight_decay,
                              weight_decay_filter=True,
                              lars_adaptation_filter=True)

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
        losses = AverageMeter('Loss', ':.4f')
        for epoch in range(self.config.num_epochs):
            # initialize meters to keep track of stats
            batch_time = AverageMeter('Data processing time (avg)', ':6.3f')
            data_time = AverageMeter('Data loading time (avg)', ':6.3f')
            progress = ProgressMeter(
                len(self.dataloader),
                [batch_time, data_time, losses],
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

                losses.update(loss.item(), self.config.batch_size)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0 or i == len(self.dataloader)-1:
                    progress.display(i)

            # warmup for the first 10 epochs
            if epoch >= self.config.warm_up_epochs:
                self.scheduler.step()

            # check if current epoch is best epoch and save model state_dict
            losses.check_best_epoch(model, epoch, self.config)

# ------------------- MoCo trainer -------------------- #


'''
Code adopted with slight changes.
Source: https://github.com/facebookresearch/moco
Date: February 17th, 2022

'''


class MoCo_trainer(object):

    def __init__(self, config, device):
        self.config = config
        self.dataloader = config.train_dataloader
        self.device = device
        self.T = 1.0

    def adjust_moco_momentum(self, epoch):
        """Adjust moco momentum based on current epoch"""
        m = 1. - 0.5 * (1. + math.cos(math.pi * epoch /
                        self.config.num_epochs)) * (1. - self.config.ema_factor)
        return m

    def adjust_learning_rate(self, epoch):
        """Decay the learning rate based on schedule"""
        lr = self.config.init_lr
        if epoch == self.config.num_epochs/2:
            lr *= 0.1
            print('modified learning rate to: ', lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, model):
        # define the optimizer

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.SGD(model.parameters(), self.config.init_lr,
                                         momentum=self.config.momentum,
                                         weight_decay=self.config.weight_decay)

        # define scaler for mixed precision
        scaler = GradScaler(enabled=self.config.fp16_precision)

        # check for optimal backend
        torch.backends.cudnn.benchmark = True

        # put model in training mode
        model.train()

        #  *** main training loop ***
        losses = AverageMeter('Loss', ':.4f')
        for epoch in range(self.config.num_epochs):
            # initialize meters to keep track of stats
            batch_time = AverageMeter('Data processing time (avg)', ':6.3f')
            data_time = AverageMeter('Data loading time (avg)', ':6.3f')
            progress = ProgressMeter(
                len(self.dataloader),
                [batch_time, data_time, losses],
                prefix="Epoch: [{}]".format(epoch))

            end = time.time()
            self.adjust_learning_rate(epoch)

            # loop through batches
            for i, (satellite, drone) in enumerate(self.dataloader):
                # send data to GPU
                drone = drone.to(self.device)
                satellite = satellite.to(self.device)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output and loss
                with autocast(enabled=self.config.fp16_precision):
                    drone_output, drone_target, sat_output, sat_target = model(
                        drone, satellite)

                    loss = self.criterion(
                        drone_output, drone_target) + self.criterion(sat_output, sat_target)

                losses.update(loss.item(), self.config.batch_size)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0 or i == len(self.dataloader)-1:
                    progress.display(i)

            # check if current epoch is best epoch and save model state_dict
            losses.check_best_epoch(model, epoch, self.config)


# ------------------- BarlowTwins trainer -------------------- #

'''
Code adopted with slight changes.
Source: https://github.com/facebookresearch/barlowtwins
Date: February 17th, 2022

'''


class BarlowTwins_trainer(object):

    def __init__(self, config, device):
        self.config = config
        self.dataloader = config.train_dataloader
        self.device = device

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def adjust_learning_rate(self, optimizer, step):
        max_steps = self.config.num_epochs * len(self.dataloader)
        warmup_steps = self.config.warm_up_epochs * len(self.dataloader)
        base_lr = self.config.batch_size / 256
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        optimizer.param_groups[0]['lr'] = lr * \
            self.config.init_lr
        optimizer.param_groups[1]['lr'] = lr * 0.0048

    def train(self, model):
        # define the optimizer
        param_weights = []
        param_biases = []
        for param in model.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{'params': param_weights}, {'params': param_biases}]
        self.optimizer = LARS(parameters, lr=self.config.init_lr, weight_decay=self.config.weight_decay,
                              weight_decay_filter=True,
                              lars_adaptation_filter=True)

        # define scaler for mixed precision
        scaler = GradScaler(enabled=self.config.fp16_precision)

        # check for optimal backend
        torch.backends.cudnn.benchmark = True

        # put model in training mode
        model.train()

        #  *** main training loop ***
        losses = AverageMeter('Loss', ':.4f')
        for epoch in range(self.config.num_epochs):
            # initialize meters to keep track of stats
            batch_time = AverageMeter('Data processing time (avg)', ':6.3f')
            data_time = AverageMeter('Data loading time (avg)', ':6.3f')
            progress = ProgressMeter(
                len(self.dataloader),
                [batch_time, data_time, losses],
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
                    pred_drone, pred_satellite = model(drone, satellite)
                    self.adjust_learning_rate(self.optimizer, i)
                    # print('output: ', pred_drone)
                    # print('bn: ', model.module.bn(pred_drone))

                    # empirical cross-correlation matrix
                    c = model.module.bn(
                        pred_drone).T @ model.module.bn(pred_satellite)
                    c = c.div_(self.config.batch_size)
                    # print('c: ', c)
                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = self.off_diagonal(c).pow_(2).sum()
                    loss = on_diag + self.config.epsilon * off_diag
                    # print('loss: ', loss)

                losses.update(loss.item(), self.config.batch_size)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0 or i == len(self.dataloader)-1:
                    progress.display(i)

            # check if current epoch is best epoch and save model state_dict
            losses.check_best_epoch(model, epoch, self.config)


# ------------------- BYOL trainer -------------------- #

'''
Code adopted with slight changes.
Source: https://github.com/sthalles/PyTorch-BYOL
Date: February 17th, 2022

'''


class BYOL_trainer(object):

    def __init__(self, config, device):
        self.config = config
        self.dataloader = config.train_dataloader
        self.device = device

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def adjust_learning_rate(self, optimizer, epoch):
        """Decays the learning rate with half-cycle cosine after warmup"""
        if epoch < self.config.warm_up_epochs:
            lr = self.config.init_lr * epoch / self.config.warm_up_epochs
        else:
            lr = self.config.init_lr * 0.5 * (1. + math.cos(math.pi * (
                epoch - self.config.warm_up_epochs) / (self.config.num_epochs - self.config.warm_up_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train(self, model):
        # define the optimizer
        self.optimizer = LARS(list(model.module.online_encoder.parameters()) + list(model.module.predictor.parameters()),
                              lr=self.config.init_lr, weight_decay=self.config.weight_decay,
                              weight_decay_filter=True,
                              lars_adaptation_filter=True)

        # define scaler for mixed precision
        scaler = GradScaler(enabled=self.config.fp16_precision)

        # check for optimal backend
        torch.backends.cudnn.benchmark = True

        # put model in training mode
        model.train()

        model.module.init_target_encoder()

        #  *** main training loop ***
        losses = AverageMeter('Loss', ':.4f')
        for epoch in range(self.config.num_epochs):
            # initialize meters to keep track of stats
            batch_time = AverageMeter('Data processing time (avg)', ':6.3f')
            data_time = AverageMeter('Data loading time (avg)', ':6.3f')
            progress = ProgressMeter(
                len(self.dataloader),
                [batch_time, data_time, losses],
                prefix="Epoch: [{}]".format(epoch))

            end = time.time()
            iters_per_epoch = len(self.dataloader)

            # loop through batches
            for i, (satellite, drone) in enumerate(self.dataloader):
                # send data to GPU
                drone = drone.to(self.device)
                satellite = satellite.to(self.device)

                # measure data loading time
                data_time.update(time.time() - end)

                lr = self.adjust_learning_rate(
                    self.optimizer, i/iters_per_epoch)

                # compute output and loss
                with autocast(enabled=self.config.fp16_precision):
                    online_pred_drone, target_proj_drone, online_pred_satellite, target_proj_satellite = model(
                        drone, satellite)
                    loss_one = self.loss_fn(
                        online_pred_drone, target_proj_satellite.detach())
                    loss_two = self.loss_fn(
                        online_pred_satellite, target_proj_drone.detach())

                    loss = (loss_one + loss_two).mean()

                losses.update(loss.item(), self.config.batch_size)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                model.module._update_target_encoder_params()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0 or i == len(self.dataloader)-1:
                    progress.display(i)

            # check if current epoch is best epoch and save model state_dict
            losses.check_best_epoch(model, epoch, self.config)


# ------------------- Triplet trainer -------------------- #

'''
Code adopted with slight changes.
Source: https://github.com/KevinMusgrave/pytorch-metric-learning
Date: February 17th, 2022

'''


class Triplet_trainer(object):
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.dataloader = config.train_dataloader

    def train(self, model):

        train_transform = transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        train_dataset = datasets.ImageFolder(
            self.config.data_store+'/triplet/'+self.config.location+'/train/'+str(self.config.patch_size), transform=train_transform)

        self.encoder_optimizer = torch.optim.Adam(
            model.module.encoder.parameters(), lr=0.0001, weight_decay=0.0001)
        self.embedder_optimizer = torch.optim.Adam(
            model.module.embedder.parameters(), lr=0.001, weight_decay=0.0001)

        # define the loss criterion
        self.loss = losses.TripletMarginLoss(margin=0.1)

        # set miner
        miner = miners.TripletMarginMiner(margin=0.1, type_of_triplets="all")

        # set sampler
        sampler = samplers.MPerClassSampler(
            train_dataset.targets, m=2, length_before_new_iter=len(train_dataset))

        # Package the above stuff into dictionaries.
        models = {"trunk": nn.DataParallel(model.module.encoder.to(self.device), self.config.gpu_ids),
                  "embedder": nn.DataParallel(model.module.embedder.to(self.device), self.config.gpu_ids)}
        optimizers = {"trunk_optimizer": self.encoder_optimizer,
                      "embedder_optimizer": self.embedder_optimizer}
        loss_funcs = {"metric_loss": self.loss}
        mining_funcs = {"tuple_miner": miner}

        trainer = trainers.MetricLossOnly(models,
                                          optimizers,
                                          self.config.batch_size,
                                          loss_funcs,
                                          mining_funcs,
                                          train_dataset,
                                          sampler=sampler,
                                          dataloader_num_workers=self.config.num_workers)

        trainer.train(num_epochs=self.config.num_epochs)

        torch.save({'model_state_dict': model.state_dict()}, self.config.dump_path +
                   '/' + self.config.model_name + '_best_epoch_'+str(self.config.patch_size)+'.pth')
