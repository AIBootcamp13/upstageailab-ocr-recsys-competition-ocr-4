from .unet import UNet
from hydra.utils import instantiate


def get_decoder_by_cfg(config):
    decoder = instantiate(config)
    return decoder
