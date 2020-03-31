import torch.nn as nn
import math
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1,stride=1)
        self.bn1 = nn.BatchNorm2d(out_channel, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channel, affine=True)
        self.bn3 = nn.BatchNorm2d(out_channel, affine=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.downsample = downsample

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = F.relu(self.bn2(self.conv2(o)))
        o = self.bn3(self.conv3(o))
        identity = self.downsample(x)
        o = F.relu(o+identity)
        o = self.maxpool(o)
        return o


class ResNet12(nn.Module):
    '''
    This ResNet12 does not contain the last fully connected layer
    Each layer's output dimention is [64, 128, 256, 512]
    '''
    def __init__(self, in_channel, image_size=28):
        super(ResNet12, self).__init__()
        self.in_channel = in_channel
        self.layer1 = self._make_layer(64)
        self.layer2 = self._make_layer(128)
        self.layer3 = self._make_layer(256)
        self.layer4 = self._make_layer(512)
        final_size = int(math.floor(image_size/(2*2*2*2)))
        self.outdim = final_size * final_size * 512

    def _make_layer(self, out_channel):
        dowmsample = nn.Sequential(
            nn.Conv2d(self.in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel)
        )
        block = ResidualBlock(self.in_channel, out_channel, dowmsample)
        self.in_channel = out_channel
        return block

    def forward(self, x):
        o = self.layer1(x)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer4(o)
        return o.view(-1, self.outdim)
