from torchvision import transforms, datasets
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from torch import nn
import torch
from utils import AverageMeter, ProgressMeter
import time
import os
import sys
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from utils import LARC, LARS
import numpy as np
import math
import torch.distributed as dist
from PIL import Image


# ------------------- SimSiam trainer -------------------- #

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

                if i % self.config.print_freq == 0:
                    progress.display(i)

            # check if current epoch is best epoch and save model state_dict
            losses.check_best_epoch(model, epoch, self.config)


# ------------------- SimCLR trainer -------------------- #

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

                if i % self.config.print_freq == 0:
                    progress.display(i)

            # warmup for the first 10 epochs
            if epoch >= self.config.warm_up_epochs:
                self.scheduler.step()

            # check if current epoch is best epoch and save model state_dict
            losses.check_best_epoch(model, epoch, self.config)

# ------------------- MoCo trainer -------------------- #


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

                if i % self.config.print_freq == 0:
                    progress.display(i)

            # check if current epoch is best epoch and save model state_dict
            losses.check_best_epoch(model, epoch, self.config)


# ------------------- BarlowTwins trainer -------------------- #

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

                if i % self.config.print_freq == 0:
                    progress.display(i)

            # check if current epoch is best epoch and save model state_dict
            losses.check_best_epoch(model, epoch, self.config)


# ------------------- BYOL trainer -------------------- #

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
        # self.optimizer = torch.optim.SGD(list(model.module.online_encoder.parameters()) + list(model.module.predictor.parameters()),
        #                                  lr=self.config.init_lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
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

                if i % self.config.print_freq == 0:
                    progress.display(i)

            # check if current epoch is best epoch and save model state_dict
            losses.check_best_epoch(model, epoch, self.config)


# ------------------- DINO trainer -------------------- #

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q *
                                 F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + \
            batch_center * (1 - self.center_momentum)


class DINO_trainer(object):

    def __init__(self, config, device):
        self.config = config
        self.dataloader = config.train_dataloader
        self.device = device

    def get_params_groups(self, model):
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    def cosine_scheduler(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(
                start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * \
            (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule

    def train(self, model):
        # define criterion
        criterion = DINOLoss(
            self.config.num_projection,
            # total number of crops = 2 global crops + local_crops_number
            2,
            0.04,
            0.04,
            self.config.warm_up_epochs,
            self.config.num_epochs,
        ).to(self.device)

        # define the optimizer
        params_groups = self.get_params_groups(model.module.encoder)
        self.optimizer = torch.optim.AdamW(params_groups)

        # define learning rate scheduler
        lr_schedule = self.cosine_scheduler(
            self.config.init_lr / 256.,  # linear scaling rule
            0,
            self.config.num_epochs, len(self.dataloader),
            warmup_epochs=self.config.warm_up_epochs,
        )
        wd_schedule = self.cosine_scheduler(
            self.config.weight_decay,
            0.4,
            self.config.num_epochs, len(self.dataloader),
        )
        momentum_schedule = self.cosine_scheduler(self.config.ema_factor, 1,
                                                  self.config.num_epochs, len(self.dataloader))

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
                # update weight decay and learning rate according to their schedule
                global_it = len(self.dataloader) * epoch + \
                    i  # global training iteration

                for j, param_group in enumerate(self.optimizer.param_groups):
                    param_group["lr"] = lr_schedule[j]
                    if j == 0:  # only the first group is regularized
                        param_group["weight_decay"] = wd_schedule[j]

                # send data to GPU
                drone = drone.to(self.device)
                satellite = satellite.to(self.device)

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output and loss
                teacher_out_drone, teacher_out_sat, student_out_drone, student_out_sat = model(
                    drone, satellite)

                loss = criterion(torch.cat((teacher_out_drone, teacher_out_sat), 0), torch.cat(
                    (student_out_drone, student_out_sat), 0), epoch)

                losses.update(loss.item(), self.config.batch_size)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                # EMA update for the teacher
                with torch.no_grad():
                    m = momentum_schedule[global_it]  # momentum parameter
                    for param_q, param_k in zip(model.module.encoder.parameters(), model.module.teacher.parameters()):
                        param_k.data.mul_(m).add_(
                            (1 - m) * param_q.detach().data)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.config.print_freq == 0:
                    progress.display(i)

            # check if current epoch is best epoch and save model state_dict
            losses.check_best_epoch(model, epoch, self.config)


# ------------------- SwAV trainer -------------------- #


class SwAV_trainer(object):

    def __init__(self, config, device):
        self.config = config
        self.dataloader = config.train_dataloader
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
        losses = AverageMeter('Loss', ':.4f')
        for epoch in range(self.config.num_epochs):
            # initialize meters to keep track of stats
            batch_time = AverageMeter('Data processing time (avg)', ':6.3f')
            data_time = AverageMeter('Data loading time (avg)', ':6.3f')
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
            for param_group in self.optimizer.param_groups:
                print('Learning rate: ', param_group["lr"])
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

            # check if current epoch is best epoch and save model state_dict
            losses.check_best_epoch(model, epoch, self.config)

    def init_queue(self):
        return torch.zeros(
            len([0, 1]),
            self.config.queue_length,
            self.config.num_features,
        ).cuda()

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        # Q is K-by-B for consistency with notations from our paper
        Q = torch.exp(out / self.config.epsilon).t()
        B = Q.shape[1]  # number of samples to assign
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


# ------------------- Triplet trainer -------------------- #


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
        models = {"trunk": model.module.encoder,
                  "embedder": model.module.embedder}
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

        torch.save({'model_state_dict': model.module.encoder.state_dict()}, self.config.dump_path +
                   '/' + self.config.model_name + '_best_epoch_'+str(self.config.patch_size)+'.pth')
