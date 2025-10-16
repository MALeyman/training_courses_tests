import time

import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
from torch.autograd import Variable


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, n_channels):
        super(MobileNet, self).__init__()
        self.layer1 = nn.Sequential(
        
            conv_bn(n_channels, 32, 1),  

            conv_dw(32, 64, 1),  

            conv_dw(64, 128, 2),  
            conv_dw(128, 128, 1),


            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )

        self.layer2 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )

        self.layer3 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg(x)

        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNet(n_channels=3)
    model.to(device)
    summary(model, input_size=(3, 256, 512))
