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

from itertools import accumulate
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 strides=[4, 8, 16, 32],
                 inner_channels=256,
                 output_channels=64,
                 bias=False):
        super(UNet, self).__init__()

        assert len(strides) == len(in_channels), "Mismatch in 'strides' and 'in_channels' lengths."

        # Parameters에 따라 UNet 구조를 동적으로 생성
        # Decoder size 계산
        upscale_factors = [strides[idx] // strides[idx - 1] for idx in range(1, len(strides))]
        outscale_factors = list(accumulate(upscale_factors, lambda x, y: x * y))

        self.upsamples = nn.ModuleList()
        for upscale in upscale_factors:
            self.upsamples.append(nn.Upsample(scale_factor=upscale, mode='nearest'))

        self.inners = nn.ModuleList()
        for in_channel in in_channels:
            self.inners.append(nn.Conv2d(in_channel, inner_channels, kernel_size=1, bias=bias))

        self.outers = nn.ModuleList()
        for outscale in reversed(outscale_factors):
            outer = nn.Sequential(nn.Conv2d(inner_channels, output_channels,
                                            kernel_size=3, padding=1, bias=bias),
                                  nn.Upsample(scale_factor=outscale, mode='nearest'))
            self.outers.append(outer)
        self.outers.append(nn.Conv2d(inner_channels, output_channels, kernel_size=3,
                                     padding=1, bias=bias))

        self.upsamples.apply(self.weights_init)
        self.inners.apply(self.weights_init)
        self.outers.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, features):
        in_features = [inner(feat) for feat, inner in zip(features, self.inners)]

        up_features = []
        up = in_features[-1]
        for i in range(len(in_features) - 1, 0, -1):
            up = self.upsamples[i - 1](up) + in_features[i - 1]
            up_features.append(up)

        out_features = [self.outers[0](in_features[-1])]
        out_features += [outer(feat) for feat, outer in zip(up_features, self.outers[1:])]

        return out_features
