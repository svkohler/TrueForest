from models.SimSiam import SimSiam
from models.SimCLR import ResNetSimCLR
from models.SwAV import ResNetSwAV, Bottleneck

import torchvision.models as models

from models.model_trainer import SimCLR_trainer, SimSiam_trainer, SwAV_trainer


def load_model(config, dataloader, device):

    # check for the right model and return it
    if config.model_name == 'SimSiam':
        base_encoder = models.__dict__[config.base_architecture]
        model = SimSiam(base_encoder=base_encoder,
                        dim=config.num_features, pred_dim=512)

        trainer = SimSiam_trainer(config, dataloader, device)
        return model, trainer

    if config.model_name == 'SimCLR':
        model = ResNetSimCLR(
            base_model=config.base_architecture, out_dim=config.num_features)
        trainer = SimCLR_trainer(config, dataloader, device)

        return model, trainer

    if config.model_name == 'SwAV':
        if config.base_architecture == 'resnet50':
            layers = [3, 4, 6, 3]
        elif config.base_architecture == 'resnet101':
            layers = [3, 4, 23, 3]

        model = ResNetSwAV(block=Bottleneck, layers=layers,
                           normalize=config.normalize, output_dim=config.num_features,
                           hidden_mlp=config.num_hidden, nmb_prototypes=3000)

        trainer = SwAV_trainer(config, dataloader, device)

        return model, trainer
