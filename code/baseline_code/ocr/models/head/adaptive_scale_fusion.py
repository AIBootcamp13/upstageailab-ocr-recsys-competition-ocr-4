import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveScaleFusion(nn.Module):
    def __init__(self, in_channels, inter_channels, out_features_num=4):
        super(AdaptiveScaleFusion, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.spatial_attention = nn.Conv2d(inter_channels, out_features_num, 1)
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, concat_x, features_list):
        conv_x = self.conv(concat_x)
        attention_out = self.spatial_attention(conv_x)
        assert len(features_list) == self.out_features_num
        x = []
        for i in range(self.out_features_num):
            x.append(attention_out[:, i:i+1] * features_list[i])
        return torch.cat(x, dim=1)
