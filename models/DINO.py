import torch.nn as nn

from utils import MultiCropWrapper
from models.ViT_DINO import DINOHead


class DINO(nn.Module):

    def __init__(self, base_encoder, config):

        self.config = config
        self.base_encoder = base_encoder()
        self.teacher = base_encoder()
        embed_dim = self.student.embed_dim

        self.student = MultiCropWrapper(self.student, DINOHead(
            embed_dim,
            self.config.projection_size,
        ))
        self.teacher = MultiCropWrapper(
            self.teacher,
            DINOHead(embed_dim, self.config.projection_size),
        )
