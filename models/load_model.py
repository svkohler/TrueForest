from models.SimSiam import SimSiam
from models.SimCLR import ResNetSimCLR
from models.BYOL import BYOL
from models.BarlowTwins import BarlowTwins
from models.MoCo import MoCo
from models.triplet import Triplet

import torchvision.models as models

from models.model_trainer import SimCLR_trainer, SimSiam_trainer, BYOL_trainer, BarlowTwins_trainer, MoCo_trainer, Triplet_trainer
from models.model_tester import Tester, Triplet_tester


def load_model(config, device):
    '''
    look up the right model, trainer, tester and return it

    '''

    if config.model_name == 'Triplet':
        base_encoder = models.__dict__[config.base_architecture]
        model = Triplet(base_encoder=base_encoder, config=config)

        trainer = Triplet_trainer(config, device)
        tester = Triplet_tester(config, device)

        return model, trainer, tester

    if config.model_name == 'SimSiam':
        base_encoder = models.__dict__[config.base_architecture]
        model = SimSiam(base_encoder=base_encoder, config=config)

        trainer = SimSiam_trainer(config, device)
        tester = Tester(config, device)

        return model, trainer, tester

    if config.model_name == 'SimCLR':
        base_encoder = models.__dict__[config.base_architecture]
        model = ResNetSimCLR(
            base_model=base_encoder, config=config)
        trainer = SimCLR_trainer(config, device)
        tester = Tester(config, device)

        return model, trainer, tester

    if config.model_name == 'BYOL':
        base_encoder = models.__dict__[config.base_architecture]
        model = BYOL(base_encoder=base_encoder, config=config)
        trainer = BYOL_trainer(config, device)
        tester = Tester(config, device)

        return model, trainer, tester

    if config.model_name == 'BarlowTwins':
        base_encoder = models.__dict__[config.base_architecture]
        model = BarlowTwins(base_encoder=base_encoder, config=config)
        trainer = BarlowTwins_trainer(config, device)
        tester = Tester(config, device)

        return model, trainer, tester

    if config.model_name == 'MoCo':

        base_encoder = models.__dict__[config.base_architecture]
        model = MoCo(base_encoder=base_encoder, config=config)
        trainer = MoCo_trainer(config, device)
        tester = Tester(config, device)

        return model, trainer, tester
