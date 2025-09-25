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


class DBPP_Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=10.0, gamma=1.0, negative_ratio=3.0, eps=1e-6):
        super(DBPP_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.negative_ratio = negative_ratio
        self.eps = eps
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

            loss = self.alpha * loss_prob + self.beta * loss_binary + self.gamma * loss_thresh
            loss_dict.update(loss_thresh=loss_thresh, loss_binary=loss_binary)
        else:
            loss = loss_prob

        return loss, loss_dict
