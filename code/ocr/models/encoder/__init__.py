from .timm_backbone import TimmBackbone
from hydra.utils import instantiate


def get_encoder_by_cfg(config):
    encoder = instantiate(config)
    return encoder
