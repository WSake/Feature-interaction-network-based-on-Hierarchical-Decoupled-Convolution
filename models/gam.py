import torch.nn as nn
import torch
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            print("downsampling")
            residual = self.downsample(x)



        out += residual
        out = self.relu(out)

        return out


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.LeakyReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm3d(int(in_channels / rate)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm3d(out_channels)
        )

        self.CBAM = BasicBlock(32,32)



    def forward(self, x):
        b, c, h, w, l = x.shape
        x_permute = x.permute(0, 2, 3, 4, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, l, c)
        x_channel_att = x_att_permute.permute(0, 4, 1, 2, 3)

        x1 = x * x_channel_att
#
        x_spatial_att = self.spatial_attention(x1).sigmoid()
        out1 = x1 * x_spatial_att




        return out1


if __name__ == '__main__':
    x = torch.randn(1, 32, 16, 16, 16)
    b, c, h, w, l = x.shape
    net = GAM_Attention(in_channels=c, out_channels=c)
    y = net(x)