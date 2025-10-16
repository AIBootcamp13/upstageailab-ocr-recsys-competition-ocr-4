from .db_loss import DBLoss
from .l1_loss import MaskL1Loss
from .bce_loss import BCELoss
from .dice_loss import DiceLoss
from hydra.utils import instantiate


def get_loss_by_cfg(config):
    loss = instantiate(config)
    return loss
