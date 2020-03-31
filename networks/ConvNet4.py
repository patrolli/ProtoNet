import torch.nn as nn
import math
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )


class ConvNet4(nn.Module):

    def __init__(self, inchannel, outchannel, image_size=28):
        super(ConvNet4, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(inchannel, outchannel),
            conv_block(outchannel, outchannel),
            conv_block(outchannel, outchannel),
            conv_block(outchannel, outchannel)
        )
        finalSize = int(math.floor(image_size/(2*2*2*2)))
        self.outdim = finalSize * finalSize * outchannel  # the feature length of an image

    def forward(self, x):
        x = self.encoder(x)
        return x.view(-1, self.outdim)

