import os
import sys
import pandas as pd
import numpy as np
import random
import math
import warnings

import torch
from torch import nn, optim
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn.functional as F

from pynvml import *


def get_memory_free_MiB(gpu_index):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2


def get_param(random_nr, range):
    return range[0] + (range[1]-range[0])*random_nr


class TrueForrestDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.satellite_rgb_dir = self.config.data_store + '/satellite_rgb/'
        self.satellite_nir_dir = self.config.data_store + '/satellite_nir/'
        self.drone_dir = self.config.data_store + '/drone/'
        self.len = self.check_len()
        self.satellite_rgb_images = sorted(os.listdir(self.satellite_rgb_dir))
        self.satellite_nir_images = sorted(os.listdir(self.satellite_nir_dir))
        self.drone_images = sorted(os.listdir(self.drone_dir))
        self.angles = [0, 90, 180, 270]
        self.contrast_range = [0.6, 2]
        self.gamma_range = [0.8, 1.3]
        self.hue_range = [-0.3, 0.4]
        self.saturation_range = [0, 2]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_satellite = ToTensor()(Image.open(
            self.satellite_rgb_dir + self.satellite_rgb_images[idx]))
        # augment with near infrared channel if activated
        if self.config.NIR:
            img_satellite_nir = ToTensor()(Image.open(
                self.satellite_nir_dir + self.satellite_nir_images[idx]))
            img_satellite = torch.cat(
                (img_satellite, img_satellite_nir), dim=0)
        img_drone = ToTensor()(Image.open(
            self.drone_dir + self.drone_images[idx]))
        # perform transformations
        if self.config.transforms.implement:
            img_satellite, img_drone = self.transform(img_satellite, img_drone)

        return img_satellite, img_drone

    def transform(self, satellite, drone):

        if self.config.transforms.hflip and torch.rand(1) < self.config.transforms.hflip_prob:
            satellite = transforms.functional.hflip(satellite)
            drone = transforms.functional.hflip(drone)

        if self.config.transforms.vflip and torch.rand(1) < self.config.transforms.vflip_prob:
            satellite = transforms.functional.vflip(satellite)
            drone = transforms.functional.vflip(drone)

        if self.config.transforms.gaussian_blur and torch.rand(1) < self.config.transforms.gaussian_blur_prob:
            blurrer = transforms.GaussianBlur(
                kernel_size=[23, 23], sigma=(0.1, 2.0))
            satellite = blurrer(satellite)
            drone = blurrer(drone)

        if self.config.transforms.contrast and torch.rand(1) < self.config.transforms.contrast_prob:
            contrast = get_param(torch.rand(1), self.contrast_range)
            satellite = transforms.functional.adjust_contrast(
                satellite, contrast)
            drone = transforms.functional.adjust_contrast(drone, contrast)

        if self.config.transforms.hue and torch.rand(1) < self.config.transforms.hue_prob:
            gamma = get_param(torch.rand(1), self.gamma_range)
            satellite = transforms.functional.adjust_gamma(satellite, gamma)
            drone = transforms.functional.adjust_gamma(drone, gamma)

        if self.config.transforms.gamma and torch.rand(1) < self.config.transforms.gamma_prob:
            hue = get_param(torch.rand(1), self.hue_range)
            satellite = transforms.functional.adjust_hue(satellite, hue)
            drone = transforms.functional.adjust_hue(drone, hue)

        if self.config.transforms.saturation and torch.rand(1) < self.config.transforms.saturation_prob:
            saturation = get_param(torch.rand(1), self.saturation_range)
            satellite = transforms.functional.adjust_saturation(
                satellite, saturation)
            drone = transforms.functional.adjust_saturation(drone, saturation)

        if self.config.transforms.rotate:
            idx = int(torch.floor(torch.rand(1)*4))
            satellite = transforms.functional.rotate(
                satellite, self.angles[idx])
            drone = transforms.functional.rotate(drone, self.angles[idx])

        if self.config.transforms.normalize:
            satellite = transforms.functional.normalize(
                satellite, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            drone = transforms.functional.normalize(
                drone, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        return satellite, drone

    def check_len(self):
        try:
            len(os.listdir(self.satellite_rgb_dir)) == len(
                os.listdir(self.drone_dir)) == len(os.listdir(self.satellite_nir_dir))
            return len(os.listdir(self.satellite_rgb_dir))
        except:
            ValueError(
                'There is not the same number of drone and satellite images.')


def check_venv(venv='mt_env'):
    if sys.prefix.split('/')[-1] != venv:
        raise ConnectionError('Not connected to correct virtual environment')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.loss_history = []
        self.best_epoch = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def check_best_epoch(self, model, epoch, config):
        self.loss_history.append(self.sum)
        if self.best_epoch is None:
            print('Found new best model.')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_history': self.loss_history,
                'config': config
            }, config.dump_path +
                '/'+config.model_name+'_best_epoch.pth')
            self.best_epoch = self.sum
        else:
            if self.best_epoch > self.sum:
                print('Found new best model.')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss_history': self.loss_history,
                    'config': config
                }, config.dump_path +
                    '/'+config.model_name+'_best_epoch.pth')
                self.best_epoch = self.sum
        self.reset()

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LARC(object):

    def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-8):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @ property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @ property
    def param_groups(self):
        return self.optim.param_groups

    @ param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = self.trust_coefficient * \
                            (param_norm) / (grad_norm +
                                            param_norm * weight_decay + self.eps)

                        # clip learning rate for LARC
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr/group['lr'], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):

        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


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


def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
