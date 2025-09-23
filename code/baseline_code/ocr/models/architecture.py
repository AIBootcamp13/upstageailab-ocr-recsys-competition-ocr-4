import torch.nn as nn
from hydra.utils import instantiate
from .encoder import get_encoder_by_cfg
from .decoder import get_decoder_by_cfg
from .head import get_head_by_cfg
from .loss import get_loss_by_cfg


class OCRModel(nn.Module):
    def __init__(self, cfg):
        super(OCRModel, self).__init__()
        self.cfg = cfg

        # 각 모듈 instantiate
        self.encoder = get_encoder_by_cfg(cfg.encoder)
        self.decoder = get_decoder_by_cfg(cfg.decoder)
        self.head = get_head_by_cfg(cfg.head)
        self.loss = get_loss_by_cfg(cfg.loss)

    def forward(self, images, return_loss=True, **kwargs):
        encoded_features = self.encoder(images)
        decoded_features = self.decoder(encoded_features)
        pred = self.head(decoded_features, return_loss)

        # Loss 계산
        if return_loss:
            loss, loss_dict = self.loss(pred, **kwargs)
            pred.update(loss=loss, loss_dict=loss_dict)

        return pred

    def get_optimizers(self):
        optimizer_config = self.cfg.optimizer
        optimizer = instantiate(optimizer_config, params=self.parameters())

        if 'scheduler' in self.cfg:
            scheduler_config = self.cfg.scheduler
            scheduler = instantiate(scheduler_config, optimizer=optimizer)
            return [optimizer], [scheduler]
        return optimizer

    def get_polygons_from_maps(self, gt, pred):
        return self.head.get_polygons_from_maps(gt, pred)
