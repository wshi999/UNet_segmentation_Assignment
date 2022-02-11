import torch
import torch.nn as nn
import torch.nn.functional as F

# import any other libraries you need below this line
import math


class twoConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(twoConvBlock, self).__init__()
        self.twoConv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=0),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(out_channel, out_channel, 3, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout2d(),
        )

    def forward(self, x):
        return self.twoConv(x)


class downStep(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(downStep, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            twoConvBlock(in_channel, out_channel),
        )

    def forward(self, x):
        return self.down(x)


class upStep(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(upStep, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channel, in_channel // 2, kernel_size=2, stride=2
        )
        self.conv = twoConvBlock(in_channel, out_channel)

    def forward(self, x, skip):
        x = self.up(x)
        crop = int((skip.shape[-1] - x.shape[-1]) / 2)
        x = self.conv(torch.cat((x, skip[:, :, crop:-crop, crop:-crop]), 1))
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv = twoConvBlock(1, 64)

        self.down1 = downStep(64, 128)
        self.down2 = downStep(128, 256)
        self.down3 = downStep(256, 512)
        self.down4 = downStep(512, 1024)

        self.up1 = upStep(1024, 512)
        self.up2 = upStep(512, 256)
        self.up3 = upStep(256, 128)
        self.up4 = upStep(128, 64)

        self.conv1 = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        skip1 = self.conv(x)

        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        x = self.down4(skip4)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        x = self.conv1(x)

        return x


class WeightMap(nn.Module):
    """
    input: label:[batch * 1 * w * h]
    """

    def __init__(self):
        super(WeightMap, self).__init__()
        self.sobel_v = (
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        ).to("cuda:0")
        self.sobel_h = (
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        ).to("cuda:0")

        # gaussian kernel from internet
        kernel_size = 17
        self.padding = (kernel_size - 1) // 2
        sigma = 5
        channels = 1
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.0
        variance = sigma ** 2.0

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
        )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        self.gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1).to("cuda:0")

    def forward(self, label):
        positive = torch.sum(label, 1)
        positive = positive / (label.shape[-1] ** 2)
        negative = 1 - positive

        w_c = label * positive + (1 - label) * negative

        label = label.float()
        v = F.conv2d(label, self.sobel_v, padding=1)
        h = F.conv2d(label, self.sobel_h, padding=1)
        edge = (torch.abs(v) + torch.abs(h)).gt(0).float()
        edge = F.conv2d(edge, self.gaussian_kernel, padding=self.padding)
        edge = torch.add(torch.mul(edge, 5), 1 * w_c) * 1
        return edge
