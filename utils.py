import pickle
import os
import numpy as np
import random

import torch
from torch import optim
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn.functional as F


def seed_all(seed):
    '''
    set seed for all random elements in the code for reproducability
    '''
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


def paths_setter(hostname, config):
    '''
    helper function to set correct paths dependent on which host machine the code is run.
    Customize here if necessary.

    '''

    if hostname == 'svkohler':
        config.data_store = "/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/data"
        config.dump_path = "/home/svkohler/OneDrive/Desktop/Masterthesis/Code/TrueForest/dump_" + \
            config.experiment_name

    elif hostname == 'spaceml1.ethz.ch':
        config.data_store = "/mnt/ds3lab-scratch/svkohler/data"
        config.dump_path = "/mnt/ds3lab-scratch/svkohler/dump_" + \
            config.experiment_name

    else:
        config.data_store = "/cluster/home/svkohler/data"
        config.dump_path = "/cluster/home/svkohler/dump_" + \
            config.experiment_name


class AccuracyCollector(object):
    '''
    Object to collect and store accuracy measurements during testing phase

    '''

    def __init__(self, num_runs):
        self.runs = 0
        self.dict = {}
        self.num_runs = num_runs

    def update(self, location, conf, run):
        if conf[0] >= 0.1:
            if location in self.dict.keys():
                self.dict[location][run] = conf
            else:
                self.dict[location] = {}
                self.dict[location][run] = conf
        elif self.runs == 0 and location not in self.dict.keys():
            self.dict[location] = {}

    def update_runs(self):
        '''
        Clean up and re-arange dictionary in case some runs were not able to converge 
        '''
        self.runs = np.min([len(self.dict[location])
                           for location in self.dict.keys()])
        for loc in self.dict.keys():
            self.dict[loc] = dict((i, value) for (
                i, (key, value)) in enumerate(self.dict[loc].items()))

    def end_statement(self):
        '''
        end statement to summarize
        '''
        for key in self.dict.keys():
            min = 1
            max = 0
            avg = []
            for k in self.dict[key]:
                v = self.dict[key][k]
                if v[0] > max:
                    max = v[0]
                if v[0] < min:
                    min = v[0]
                avg.append(v[0])
            avg = np.mean(avg)
            print(
                f'Summary for {key}: Avg. acuracy: {avg} \t Max. accuracy: {max} \t Min. accuracy: {min}')


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
        self.loss_history.append(self.avg)
        if self.best_epoch is None:
            print('Found new best model.')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_history': self.loss_history,
                'config': config
            }, config.dump_path +
                '/'+config.model_name+'_best_epoch_'+str(config.patch_size)+'.pth')
            self.best_epoch = self.avg
        else:
            if self.best_epoch > self.avg:
                print('Found new best model.')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss_history': self.loss_history,
                    'config': config
                }, config.dump_path +
                    '/'+config.model_name+'_best_epoch_'+str(config.patch_size)+'.pth')
                self.best_epoch = self.avg
        self.reset()

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    '''
    displays training progress
    '''

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


class TrueForestDataset(Dataset):
    '''
    dataset to train TrueForest models
    '''

    def __init__(self, config, mode, transform=True):
        self.config = config
        self.mode = mode
        self.trans = transform
        # construct datapaths
        self.satellite_rgb_dir = self.config.data_store + '/satellite_rgb/' + \
            config.location + '/' + self.mode + \
            '/' + str(config.patch_size) + '/'
        self.drone_dir = self.config.data_store + '/drone/' + \
            config.location + '/' + self.mode + \
            '/' + str(config.patch_size) + '/'

        self.len = self.check_len()
        self.satellite_rgb_images = sorted(os.listdir(self.satellite_rgb_dir))
        self.drone_images = sorted(os.listdir(self.drone_dir))

        # transformation parameter
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

        img_drone = ToTensor()(Image.open(
            self.drone_dir + self.drone_images[idx]))
        # perform transformations
        if self.trans:
            img_satellite, img_drone = self.transform(img_satellite, img_drone)

        if self.config.normalize:
            img_satellite, img_drone = self.normalize(img_satellite, img_drone)

        return img_satellite, img_drone

    def normalize(self, satellite, drone):
        ''' normalize the input images '''
        satellite = transforms.functional.normalize(
            satellite, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        drone = transforms.functional.normalize(
            drone, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        return satellite, drone

    def transform(self, satellite, drone):
        ''' transform satellite and drone images the same way '''

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
            contrast = self.get_param(torch.rand(1), self.contrast_range)
            satellite = transforms.functional.adjust_contrast(
                satellite, contrast)
            drone = transforms.functional.adjust_contrast(drone, contrast)

        if self.config.transforms.hue and torch.rand(1) < self.config.transforms.hue_prob:
            gamma = self.get_param(torch.rand(1), self.gamma_range)
            satellite = transforms.functional.adjust_gamma(satellite, gamma)
            drone = transforms.functional.adjust_gamma(drone, gamma)

        if self.config.transforms.gamma and torch.rand(1) < self.config.transforms.gamma_prob:
            hue = self.get_param(torch.rand(1), self.hue_range)
            satellite = transforms.functional.adjust_hue(satellite, hue)
            drone = transforms.functional.adjust_hue(drone, hue)

        if self.config.transforms.saturation and torch.rand(1) < self.config.transforms.saturation_prob:
            saturation = self.get_param(torch.rand(1), self.saturation_range)
            satellite = transforms.functional.adjust_saturation(
                satellite, saturation)
            drone = transforms.functional.adjust_saturation(drone, saturation)

        if self.config.transforms.rotate:
            idx = int(torch.floor(torch.rand(1)*4))
            satellite = transforms.functional.rotate(
                satellite, self.angles[idx])
            drone = transforms.functional.rotate(drone, self.angles[idx])

        return satellite, drone

    def check_len(self):
        ''' check if each path contains the same number of images '''

        if len(os.listdir(self.satellite_rgb_dir)) == len(os.listdir(self.drone_dir)):
            return len(os.listdir(self.satellite_rgb_dir))
        else:
            raise ValueError(
                'There is not the same number of drone and satellite images.')

    def get_param(self, random_nr, range):
        '''
        helper function to get transform parameter in the correct range
        '''
        return range[0] + (range[1]-range[0])*random_nr


def create_datasets(config):
    ''' wrapper function to create datasets. test datasets are organized as a dictionary for the different locations '''

    test_dataset = {}
    if config.location == 'all':
        config.location = 'Central_Valley'
        train_dataset = TrueForestDataset(config, mode='train')
        for loc in ['Central_Valley', 'Florida', 'Louisiana', 'Tennessee', 'Phoenix']:
            config.location = loc
            test_dataset[loc] = TrueForestDataset(
                config, mode='test', transform=False)
        config.location = 'all'
    else:
        swap = config.location
        config.location = 'Central_Valley'
        train_dataset = TrueForestDataset(config, mode='train')
        config.location = swap
        test_dataset[config.location] = TrueForestDataset(
            config, mode='test', transform=False)

    return train_dataset, test_dataset


def create_dataloader(config):
    ''' wrapper function to create the corresponding dataloaders to the datasets. Here again the test dataloaders are organized as a dict'''

    train_dataset, test_dataset = create_datasets(config)

    test_dataloader = {}

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle,
                                                   num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)

    for ds in test_dataset.keys():
        test_dataloader[ds] = torch.utils.data.DataLoader(test_dataset[ds], batch_size=config.batch_size, shuffle=False,
                                                          num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=False)

    return train_dataloader, test_dataloader


class LARC(object):
    '''
    optimizer used in the SwAV model
    '''

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
    '''
    custom optimizer used in SimCLR, BarlowTwins, BYOL

    '''

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


def process_data(data, config, mode, perc_pos=0.9):
    '''
    input: data in the format (samples * 2xfeatures)
    in addition to positive samples (which are given by definition) create negative ones by randomly
    combining satellite features (1st half) and drone features (2nd half) of different samples

    possible to not use all of the data in order to get divers datasets to train classifiers
    '''

    # if in test mode use all data
    if mode == 'test':
        perc_pos = 1

    # create index to randomly select positive samples
    idx = np.random.choice(data.shape[0], size=int(
        data.shape[0]*perc_pos), replace=False)
    data = data[idx, :]
    # create corresponding positive labels
    pos_labels = np.ones(len(idx), dtype=np.int8)

    # shuffle the data to get random pairs during production of negative samples
    data_shuffled = data.copy()
    np.random.shuffle(data_shuffled)
    negative_samples = produce_negative_samples(data_shuffled)
    # create corresponding negative labels
    neg_labels = np.zeros(len(negative_samples), dtype=np.int8)

    return np.concatenate((data, negative_samples), axis=0), np.concatenate((pos_labels, neg_labels), axis=0, dtype=np.int8)


def produce_negative_samples(data):
    ''' helper function to produce negative samples. input are shuffled data'''

    # make copy
    data_copy = data.copy()
    # shift first row to the end. Combined with the fact that we get shuffled dataset this leads
    # no accidental positive pairs which are declared as negative ones
    data_copy = data_copy[1:, :]
    data_copy = np.append(data_copy, [data[0, :]], axis=0)
    # return first half of columns from original shuffled dataset and second half of columns from the shifted one
    return np.concatenate((data[:, :int(data.shape[1]/2)], data_copy[:, int(data_copy.shape[1]/2):]), axis=1)


def compute_similarities(train_embeddings, test_embeddings, config):
    '''
    function to produce complete set of similarities

    '''
    similarity_embeddings(train_embeddings, config)
    for emb in test_embeddings.keys():
        similarity_embeddings(
            test_embeddings[emb], config, loc=emb, mode='test')


def similarity_embeddings(data, config, loc='Central_Valley', mode='train', verbose=True):
    ''' 
    function returns multiple similarity statistics for a given dataset of embeddings.
    To account for possible outliers the statistics are also calculated on a trimmed basis.

    '''

    data_shuffled = data.copy()
    np.random.shuffle(data_shuffled)
    negatives = produce_negative_samples(data_shuffled)

    pos_dot = np.apply_along_axis(dot_sim, 1, data)
    pos_cos = np.apply_along_axis(cos_sim, 1, data)
    pos_mse = np.apply_along_axis(mse, 1, data)
    neg_dot = np.apply_along_axis(dot_sim, 1, negatives)
    neg_cos = np.apply_along_axis(cos_sim, 1, negatives)
    neg_mse = np.apply_along_axis(mse, 1, negatives)

    results = {
        'positive': {
            'dot': {
                'standard': {
                    'mean': np.mean(pos_dot),
                    'median': np.median(pos_dot),
                    'std': np.std(pos_dot)
                },
                'trimmed': {
                    'mean': trim_stat(pos_dot),
                    'std': trim_stat(pos_dot, stat='std')
                }
            },
            'cos': {
                'standard': {
                    'mean': np.mean(pos_cos),
                    'median': np.median(pos_cos),
                    'std': np.std(pos_cos)
                },
                'trimmed': {
                    'mean': trim_stat(pos_cos),
                    'std': trim_stat(pos_cos, stat='std')
                }
            },
            'mse': {
                'standard': {
                    'mean': np.mean(pos_mse),
                    'median': np.median(pos_mse),
                    'std': np.std(pos_mse)
                },
                'trimmed': {
                    'mean': trim_stat(pos_mse),
                    'std': trim_stat(pos_mse, stat='std')
                }
            }
        },
        'negative': {
            'dot': {
                'standard': {
                    'mean': np.mean(neg_dot),
                    'median': np.median(neg_dot),
                    'std': np.std(neg_dot)
                },
                'trimmed': {
                    'mean': trim_stat(neg_dot),
                    'std': trim_stat(neg_dot, stat='std')
                }
            },
            'cos': {
                'standard': {
                    'mean': np.mean(neg_cos),
                    'median': np.median(neg_cos),
                    'std': np.std(neg_cos)
                },
                'trimmed': {
                    'mean': trim_stat(neg_cos),
                    'std': trim_stat(neg_cos, stat='std')
                }
            },
            'mse': {
                'standard': {
                    'mean': np.mean(neg_mse),
                    'median': np.median(neg_mse),
                    'std': np.std(neg_mse)
                },
                'trimmed': {
                    'mean': trim_stat(neg_mse),
                    'std': trim_stat(neg_mse, stat='std')
                }
            }
        }
    }

    if not os.path.exists(config.dump_path + '/similarities/'):
        os.makedirs(config.dump_path + '/similarities/')

    with open(config.dump_path + '/similarities/'+config.model_name+'_similarities_'+mode+'_'+str(config.patch_size)+'_'+loc+'.json', 'wb') as fp:
        pickle.dump(results, fp)

    if verbose:
        print(
            f'{mode} similarities of embeddings in {loc} for patch size {config.patch_size}: ')
        print(results)


'''
Below: helper fucntions for similarity statistics

'''


def dot_sim(row):
    return np.dot(row[:int(len(row)/2)], row[int(len(row)/2):])


def cos_sim(row):
    return np.dot(row[:int(len(row)/2)], row[int(len(row)/2):])/(np.linalg.norm(row[:int(len(row)/2)])*np.linalg.norm(row[int(len(row)/2):]))


def mse(row):
    return np.square(np.subtract(row[:int(len(row)/2)], row[int(len(row)/2):])).mean()


def trim_stat(arr, upper_quantile=0.99, lower_quantile=0.01, stat='mean'):
    upper_q = np.quantile(arr, upper_quantile)
    lower_q = np.quantile(arr, lower_quantile)

    arr_new = [x for x in arr if upper_q > x > lower_q]

    if stat == 'mean':
        return np.array(arr_new).mean()

    if stat == 'std':
        return np.array(arr_new).std()

    print('Error: stat not implemented.')
