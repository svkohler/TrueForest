from models.SimSiam import SimSiam

import torchvision.models as models

from models.model_trainer import SimSiam_trainer


def load_model(config, dataloader, device):

    # check for the right model and return it
    if config.model_name == 'SimSiam':
        base_encoder = models.__dict__[config.base_architecture]
        model = SimSiam(base_encoder=base_encoder,
                        dim=config.num_features, pred_dim=512)

        trainer = SimSiam_trainer(config, dataloader, device)
        return model, trainer
