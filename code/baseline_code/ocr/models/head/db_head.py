'''
*****************************************************************************************
* Modified from https://github.com/MhLiao/DB/blob/master/decoders/seg_detector.py
*
* 참고 논문:
* Real-time Scene Text Detection with Differentiable Binarization
* https://arxiv.org/pdf/1911.08947.pdf
*
* 참고 Repository:
* https://github.com/MhLiao/DB/
*****************************************************************************************
'''

import math
import torch
import torch.nn as nn
from .db_postprocess import DBPostProcessor
from collections import OrderedDict


class DBHead(nn.Module):
    def __init__(self, in_channels=256, upscale=4, k=50, bias=False, smooth=False,
                 postprocess=None):
        super(DBHead, self).__init__()
        assert postprocess is not None, "postprocess should not be None for DBHead"

        self.postprocess = DBPostProcessor(**postprocess)
        self.in_channels = in_channels
        self.inner_channels = in_channels // 4
        self.k = k

        # Feature embedding을 Upscale에 따라 확장
        self.upscale = int(math.log2(upscale))

        # Output of Probability map
        # Upscale에 따라 ConvTranspose2d Layer를 동적으로 생성
        binarize_layers = [nn.Conv2d(self.in_channels, self.inner_channels, kernel_size=3,
                                     padding=1, bias=bias),
                           nn.BatchNorm2d(self.inner_channels),
                           nn.ReLU(inplace=True)]
        for i in range(self.upscale):
            if i == self.upscale - 1:
                binarize_layers.append(
                    nn.ConvTranspose2d(self.inner_channels, 1, 2, 2)
                )
            else:
                binarize_layers.append(nn.ConvTranspose2d(self.inner_channels,
                                                          self.inner_channels, 2, 2))
                binarize_layers.append(nn.BatchNorm2d(self.inner_channels))
                binarize_layers.append(nn.ReLU(inplace=True))
        binarize_layers.append(nn.Sigmoid())
        self.binarize = nn.Sequential(*binarize_layers)
        self.binarize.apply(self.weights_init)

        # Output of Threshold map
        self.thresh = self._init_thresh(smooth=smooth, bias=bias)
        self.thresh.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, smooth=False, bias=False):
        # Upscale에 따라 Upsample Layer를 동적으로 생성
        thresh_layers = [nn.Conv2d(self.in_channels, self.inner_channels, kernel_size=3,
                                   padding=1, bias=bias),
                         nn.BatchNorm2d(self.inner_channels),
                         nn.ReLU(inplace=True)]
        for i in range(self.upscale):
            if i == self.upscale - 1:
                thresh_layers.append(self._init_upsample(self.inner_channels, out_channels=1,
                                                         smooth=smooth, bias=bias))
            else:
                thresh_layers.append(self._init_upsample(self.inner_channels, self.inner_channels,
                                                         smooth=smooth, bias=bias))
                thresh_layers.append(nn.BatchNorm2d(self.inner_channels))
                thresh_layers.append(nn.ReLU(inplace=True))
        thresh_layers.append(nn.Sigmoid())
        thresh = nn.Sequential(*thresh_layers)

        return thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        # Smooth 가 True인 경우, ConvTranspose2d 대신 Upsample을 사용
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, padding=1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=0, bias=True))
            return nn.Sequential(*module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def _step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, features, return_loss=True):
        # Input feature concat
        fuse = torch.cat(features, dim=1)

        # Probability map
        binary = self.binarize(fuse)

        if return_loss:
            # Threshold map
            thresh = self.thresh(fuse)

            # Approximate Binary map
            thresh_binary = self._step_function(binary, thresh)
            result = OrderedDict(prob_maps=binary, thresh_maps=thresh, binary_maps=thresh_binary)
        else:
            # Probability map only - Inference mode
            result = OrderedDict(prob_maps=binary)

        return result

    def get_polygons_from_maps(self, gt, pred):
        return self.postprocess.represent(gt, pred)
