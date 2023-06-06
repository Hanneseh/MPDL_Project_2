import torch
import torch.nn as nn
import torch.nn.functional as F

# this is an uncheckt chatGPT implementation


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out

class NIXNet(nn.Module):
    def __init__(self):
        super(NIXNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            ResNetBlock(3, 128),
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 512, stride=2)
        )
        self.fusion_module1 = FusionModule(128+256+512, 512)
        self.fusion_module2 = FusionModule(128+256+512, 512)
        self.fusion_module3 = FusionModule(512*2, 512)
        self.final_conv = nn.Conv2d(512*3, 1, kernel_size=1)

    def forward(self, x):
        xi = x
        ri = x - self.feature_extraction(x)
        xi_features = self.feature_extraction(xi)
        ri_features = self.feature_extraction(ri)
        xi_fused = self.fusion_module1(xi_features)
        ri_fused = self.fusion_module2(ri_features)
        fused = torch.cat([xi_fused, ri_fused], dim=1)
        fused = self.fusion_module3(fused)
        out = self.final_conv(fused)
        out = torch.sigmoid(out)
        return out
