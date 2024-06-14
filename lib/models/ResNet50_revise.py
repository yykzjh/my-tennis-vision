# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/06/14 06:10
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import torch
import torch.nn as nn
from torchvision import models

resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)


class ResNet50Revise(nn.Module):

    def __init__(self, out_channel=19):
        super(ResNet50Revise, self).__init__()
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dconv4 = nn.Conv2d(2048 + 1024, 1024, 3, 1, 1)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dconv3 = nn.Conv2d(1024 + 512, 512, 3, 1, 1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dconv2 = nn.Conv2d(512 + 256, 256, 3, 1, 1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dconv1 = nn.Conv2d(256 + 64, 64, 3, 1, 1)

        self.out_up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.out_conv = nn.Conv2d(64, out_channel, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.bn1(x1)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer1(x2)

        x3 = self.layer2(x2)

        x4 = self.layer3(x3)

        x5 = self.layer4(x4)

        d4 = self.up4(x5)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dconv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dconv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dconv2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dconv1(d1)

        out = self.out_up(d1)
        out = self.out_conv(out)

        return out


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = torch.rand((4, 3, 384, 640)).to(device)

    model = ResNet50Revise().to(device)

    checkpoint = torch.load(r"./pretrain/keypoints_model.pth", map_location=device)
    print(checkpoint.keys())

    print(model.state_dict().keys())

    output = model(image)
