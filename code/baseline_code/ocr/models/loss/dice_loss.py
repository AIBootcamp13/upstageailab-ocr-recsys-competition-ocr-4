'''
*****************************************************************************************
* Modified from https://github.com/MhLiao/DB/blob/master/decoders/dice_loss.py
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


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, 1, H, W)
        weights: (N, 1, H, W)
        '''
        assert pred.dim() == 4, pred.dim()
        if mask is None:
            mask = torch.ones_like(gt).to(device=gt.device)
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape, f"{pred.shape}, {mask.shape}"
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss
