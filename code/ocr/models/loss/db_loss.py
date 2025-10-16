'''
*****************************************************************************************
* Modified from https://github.com/MhLiao/DB/blob/master/decoders/seg_detector_loss.py#L173
*
* 참고 논문:
* Real-time Scene Text Detection with Differentiable Binarization
* https://arxiv.org/pdf/1911.08947.pdf
*
* 참고 Repository:
* https://github.com/MhLiao/DB/
*****************************************************************************************
'''

from collections import OrderedDict
import torch.nn as nn
from .bce_loss import BCELoss
from .l1_loss import MaskL1Loss
from .dice_loss import DiceLoss


class DBLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6,
                 prob_map_loss_weight=5.0,
                 thresh_map_loss_weight=10.0,
                 binary_map_loss_weight=1.0,
                 ):
        super(DBLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.prob_map_loss_weight = prob_map_loss_weight
        self.thresh_map_loss_weight = thresh_map_loss_weight
        self.binary_map_loss_weight = binary_map_loss_weight

        self.dice_loss = DiceLoss(self.eps)
        self.bce_loss = BCELoss(self.negative_ratio, self.eps)
        self.l1_loss = MaskL1Loss()

    def forward(self, pred, **kwargs):
        pred_prob = pred['prob_maps']
        pred_thresh = pred.get('thresh_maps', None)
        pred_binary = pred.get('binary_maps', None)

        gt_prob_maps = kwargs.get('prob_maps', None)
        gt_thresh_maps = kwargs.get('thresh_maps', None)
        gt_prob_mask = kwargs.get('prob_mask', None)
        gt_thresh_mask = kwargs.get('thresh_mask', None)

        loss_prob = self.bce_loss(pred_prob, gt_prob_maps, gt_prob_mask)
        loss_dict = OrderedDict(loss_prob=loss_prob)
        if pred_thresh is not None:
            loss_thresh = self.l1_loss(pred_thresh, gt_thresh_maps, gt_thresh_mask)
            loss_binary = self.dice_loss(pred_binary, gt_prob_maps, gt_prob_mask)

            loss = (self.prob_map_loss_weight * loss_prob +
                    self.thresh_map_loss_weight * loss_thresh +
                    self.binary_map_loss_weight * loss_binary)
            loss_dict.update(loss_thresh=loss_thresh, loss_binary=loss_binary)
        else:
            loss = loss_prob

        return loss, loss_dict
