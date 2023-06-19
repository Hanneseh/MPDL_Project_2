import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph
from torch import cat
from torchinfo import summary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the input and output channels are different, adjust using a 1x1 convolution
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out

    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Implements a single ConvBlock for the NIX-Net

        Args:
            in_channels (int): Specifies the input channels for the ConvBlock
            out_channels (int): Specifies the output channels for the ConvBlock
        """
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same"),
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        
    def forward(self, x):
        """Implements the forward methode of nn.Module

        Args:
            x (torch.tensor): Single tensor of shape BATCHxIN_CHANNELSxWIDTHxHEIGHT

        Returns:
            torch.tensor: Prediction of ConvBlock
        """
        out = self.conv1(x)
        out = self.conv2(out)
        return out
    

class Upsample(nn.Module):
    def __init__(self, out_size, in_channels, out_channels):
        """Implements the Upsample 1x1 operation of the NIX-Net

        Args:
            out_size ([int, int]): Output width and height of the Tensor
            in_channels (int): Specifies the input channels of the Upsample operation
            out_channels (int): Specifies the output channels fof the Upsample operation
        """
        super(Upsample, self).__init__()
        self.out_size = out_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding="same")

    def forward(self, x):
        """Implements the forward methode of nn.Module

        Args:
            x (torch.tensor): Single tensor of shape BATCHxIN_CHANNELSxWIDTHxHEIGHT

        Returns:
            torch.tensor: Prediction of Upsample
        """
        x = F.interpolate(x, size=self.out_size, mode="bilinear")
        x = self.conv(x)
        return x


class Stride(nn.Module):
    def __init__(self, out_size, in_channels, out_channels):
        """Implements the stride-2 3x3 operation of the NIX-Net

        Args:
            out_size ([int, int]): Output width and height of the Tensor
            in_channels (int): Specifies the input channels of the Stride operation
            out_channels (int): Specifies the output channels fof the Stride operation
        """
        super(Stride, self).__init__()
        self.out_size = out_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """Implements the forward methode of nn.Module

        Args:
            x (torch.tensor): Single tensor of shape BATCHxIN_CHANNELSxWIDTHxHEIGHT

        Returns:
            torch.tensor: Prediction of Stride
        """
        x = F.interpolate(x, size=self.out_size, mode="bilinear")
        x = self.conv(x)
        return x


class FusionModule(nn.Module):
    def __init__(self, width_1, height_1, width_2, height_2, width_3, height_3):
        super(FusionModule, self).__init__()
        self.upsample_1 = Upsample([width_1, height_1], 256, 128)
        self.upsample_2 = nn.Sequential(Upsample([width_2, height_2], 512, 256), 
                                        Upsample([width_1, height_1], 256, 128)) 

        self.stride_1 = Stride([width_2*2, height_2*2], 128, 256)
        self.upsample_3 = Upsample([width_2, height_2], 512, 256)

        self.stride_2 = nn.Sequential(Stride([width_2, height_2], 128, 256), 
                                      Stride([width_3*2, height_3*2], 256, 512))
        self.stride_3 = Stride([int(width_3*2), int(height_3*2)], 256, 512)
    
    def forward(self, feature_1, feature_2, feature_3):
        feature_1_out = feature_1 + self.upsample_1(feature_2) + self.upsample_2(feature_3)
        feature_2_out = self.stride_1(feature_1) + feature_2 + self.upsample_3(feature_3)
        feature_3_out = self.stride_2(feature_1)+ self.stride_3(feature_2) + feature_3
        return feature_1_out, feature_2_out, feature_3_out


class NIX(nn.Module):
    def __init__(self, img_width, img_height):
        super(NIX, self).__init__()
        self.img_width_1 = img_width
        self.img_width_2 = int(np.floor((self.img_width_1-1)/2 + 1))
        self.img_width_3 = int(np.floor((self.img_width_2-1)/2 + 1))
        self.img_width_4 = int(np.floor((self.img_width_3-1)/2 + 1))

        self.img_height_1 = img_height
        self.img_height_2 = int(np.floor((self.img_height_1-1)/2 + 1))
        self.img_height_3 = int(np.floor((self.img_height_2-1)/2 + 1))
        self.img_height_4 = int(np.floor((self.img_height_3-1)/2 + 1))

        self.res_1_x = ResidualBlock(3, 128, stride=2)
        self.res_2_x = ResidualBlock(128, 256, stride=2)
        self.res_3_x = ResidualBlock(256, 512, stride=2)
        self.res_1_r = ResidualBlock(3, 128, stride=2)
        self.res_2_r = ResidualBlock(128, 256, stride=2)
        self.res_3_r = ResidualBlock(256, 512, stride=2)

        self.fusion_1 = FusionModule(self.img_width_2, self.img_height_2, 
                                     self.img_width_3, self.img_height_3, 
                                     self.img_width_4, self.img_height_4)
        self.fusion_2 = FusionModule(self.img_width_2, self.img_height_2, 
                                     self.img_width_3, self.img_height_3, 
                                     self.img_width_4, self.img_height_4)
        self.fusion_3 = FusionModule(self.img_width_2, self.img_height_2, 
                                     self.img_width_3, self.img_height_3, 
                                     self.img_width_4, self.img_height_4)

        self.conv_1 = ConvBlock(256, 128)
        self.conv_2 = ConvBlock(512, 256)
        self.conv_3 = ConvBlock(1024, 512)
        self.conv_4 = ConvBlock(384, 1)

        self.upsample_1 = Upsample([self.img_width_2, self.img_height_2], 256, 128)
        self.upsample_2 = nn.Sequential(Upsample([self.img_width_3, self.img_height_3], 512, 256), 
                                        Upsample([self.img_width_2, self.img_height_2], 256, 128)) 
        self.upsample_3 = Upsample([self.img_width_1, self.img_height_1], 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, r):
        #Feature Extraction
        x = self.res_1_x(x)
        x_1 = x.clone()
        x = self.res_2_x(x)
        x_2 = x.clone()
        x = self.res_3_x(x)
        x_3 = x.clone()

        r = self.res_1_r(r)
        r_1 = r.clone()
        r = self.res_2_r(r)
        r_2 = r.clone()
        r = self.res_3_r(r)
        r_3 = r.clone()

        #Multi-Scale Cross Function
        feature_1_x, feature_2_x, feature_3_x = self.fusion_1(x_1, x_2, x_3)
        feature_1_r, feature_2_r, feature_3_r = self.fusion_2(r_1, r_2, r_3)

        feature_1 = self.conv_1(cat((feature_1_x, feature_1_r), dim=1))
        feature_2 = self.conv_2(cat((feature_2_x, feature_2_r), dim=1))
        feature_3 = self.conv_3(cat((feature_3_x, feature_3_r), dim=1))
        
        feature_1, feature_2, feature_3 = self.fusion_3(feature_1, feature_2, feature_3)

        #Mask Detection
        out = cat((feature_1, self.upsample_1(feature_2), self.upsample_2(feature_3)), dim=1)
        out = self.conv_4(out)
        out = self.upsample_3(out)
        #out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    img_width, img_height = 512, 512
    nix = NIX(img_width, img_height)
    print(summary(nix, [(3, img_width, img_height), (3, img_width, img_height)], batch_dim = 0, 
                  col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), 
                  verbose=0))
    