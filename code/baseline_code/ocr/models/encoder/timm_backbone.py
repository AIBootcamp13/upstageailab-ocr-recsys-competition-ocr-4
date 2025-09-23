import torch.nn as nn
import timm


class TimmBackbone(nn.Module):
    def __init__(self, model_name='resnet18', select_features=[1, 2, 3, 4], pretrained=True):
        super(TimmBackbone, self).__init__()
        # Timm Backbone 모델을 자유롭게 사용
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        # Decoder에 연결하려는 Feature를 선택
        self.select_features = select_features

    def forward(self, x):
        features = self.model(x)
        return [features[i] for i in self.select_features]
