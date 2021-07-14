import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torchvision as tv
import operator
import data as dataLoader



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv.in =  nn.Conv2d(in_channels, out_channels, stride=stride, bias=False, kernel_size=1)
        if in_channels != out_channels:
            padding = 3
        else:
            padding = 2

        self.conv_1 = nn.Conv2d(in_channels, out_channels, stride=stride, padding=padding, bias=False, kernel_size=3)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, bias=False, kernel_size=3)

        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.conv_in(x)
        x_1 = self.conv_1(x)
        x_2 = self.batch_norm_1(x_1)

        x_3 = self.relu(x_2)
        x_4 = self.conv_2(x_3)

        x_5 = self.batch_norm_2(x_4)
        # ! we are adding residual
        return x_5 + residual

class ResNet(nn.Module):
    def __init__(self, out_channels=3, stride=2):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, stride=2, padding=3, kernel_size=7)
        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.resblock_1 = ResBlock(64, 64, 1)
        self.resblock_2 = ResBlock(64, 128, 2)
        self.resblock_3 = ResBlock(128, 256, 2)
        self.resblock_4 = ResBlock(256, 512, 2)
        self.global_avg_pool = nn.AvgPool2d(4)
        self.flatten_layer = nn.Flatten()
        self.fully_connec_linear = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv_in(x)
        x_1 = self.relu(self.batch_norm(x))
        x_2 = self.max_pool(x_1)
        x_3 = self.resblock_1(x_2)
        x_4 = self.resblock_2(x_3)
        x_5 = self.resblock_3(x_4)
        x_6 = self.resblock_4(x_5)
        x_7 = self.global_avg_pool(x_6)
        x_8 = self.flatten_layer(x_7)
        x_9 = self.fully_connec_linear(x_8)
        x_sigmoid = torch.sigmoid(x_9)
        return x_sigmoid
