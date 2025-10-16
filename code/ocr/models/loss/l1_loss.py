'''
*****************************************************************************************
* Modified from https://github.com/MhLiao/DB/blob/master/decoders/l1_loss.py
*
* 참고 논문:
* Real-time Scene Text Detection with Differentiable Binarization
* https://arxiv.org/pdf/1911.08947.pdf
*
* 참고 Repository:
* https://github.com/MhLiao/DB/
*****************************************************************************************
'''

import torch
import torch.nn as nn


class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, gt, mask=None):
        if mask is None:
            mask = torch.ones_like(gt).to(device=gt.device)
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum
        else:
            loss = (torch.abs(pred - gt) * mask).sum() / mask_sum
            return loss
