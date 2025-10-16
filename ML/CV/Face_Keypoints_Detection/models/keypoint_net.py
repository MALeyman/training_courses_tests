""" 
Автор: Лейман Максим  

Дата создания: 18.06.2025
"""


import segmentation_models_pytorch as smp
import torch.nn as nn

class KeypointNet(nn.Module):
    def __init__(self, encoder_name='resnet34', num_keypoints=68):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights='imagenet'
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.encoder.out_channels[-1], num_keypoints * 2)

    def forward(self, x):
        features = self.encoder(x)[-1]
        pooled = self.pool(features).flatten(1)
        out = self.fc(pooled)
        return out






