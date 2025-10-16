'''
*****************************************************************************************
* Modified from https://github.com/MhLiao/DB/blob/master/decoders/balance_cross_entropy_loss.py
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


class BCELoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BCELoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, gt, mask=None):
        if mask is None:
            mask = torch.ones_like(gt).to(device=gt.device)
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = ((positive_loss.sum() + negative_loss.sum()) /
                        (positive_count + negative_count + self.eps))

        return balance_loss
