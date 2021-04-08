from torch import nn


class repConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.conv = nn.Conv2d(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv1x1 = nn.Conv2d(self, in_channels, out_channels, 1, stride, padding, dilation, groups, bias)

    def forward(self, x):
        conv_output = self.conv(x)
        conv1x1_output = self.conv1x1(x)
        output = conv_output + conv1x1_output + x
        return output

class repConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.conv = nn.Conv3d(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv1x1 = nn.Conv3d(self, in_channels, out_channels, 1, stride, padding, dilation, groups, bias)

    def forward(self, x):
        conv_output = self.conv(x)
        conv1x1_output = self.conv1x1(x)
        output = conv_output + conv1x1_output + x
        return output
