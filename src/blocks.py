import warnings # ignore future warnings for now
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path # Path library

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch # pytorch library
import torch.nn as nn # neural network module
# import torch.optim as optim # optimization module
# from torch.utils.data import DataLoader # data loader module
# import torchvision.transforms as transforms # image transforms module
#
# from matplotlib import pyplot as plt # plotting library
# from IPython.core.display import SVG # display SVG images in Jupyter
# from tqdm import tqdm, trange # progress bar library


# set seaborn style to no grid and white background
# import seaborn as sns # seaborn library for plotting figures
# sns.set(style='white', rc={'axes.grid': False})


class ConvEncoder(nn.Module):
    """Learnable position embedding + initial projection."""
    def __init__(self, num_channels=64, kernel_size=5, strides=1, pooling=2):
        super(ConvEncoder, self).__init__()

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.pooling = pooling

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size, stride=strides, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size, stride=strides, padding=kernel_size//2, bias=False)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size, stride=strides, padding=kernel_size//2, bias=False)
        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size, stride=strides, padding=kernel_size//2, bias=False)

        # Pooling layers
        self.pool = nn.MaxPool2d(pooling, stride=pooling)

        # Normalization layers
        self.norm1 = nn.GroupNorm(16, num_channels)
        self.norm2 = nn.GroupNorm(16, num_channels)
        self.norm3 = nn.GroupNorm(4, num_channels)
        self.norm4 = nn.GroupNorm(4, num_channels)

        # Activation layers
        self.activation = nn.ReLU()


    def forward(self, tokens):
        # First convolutional block
        x = self.conv1(tokens)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.pool(x)
        print(x.shape)

        # Second convolutional block
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.pool(x)
        print(x.shape)
        # Third convolutional block
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.pool(x)
        print(x.shape)
        # Final convolutional block
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.activation(x)
        print(x.shape)
        # Reshape and concatenate operations
        return x


class ConvDecoder(nn.Module):
    """Convert transformer output to dose."""
    def __init__(self, num_channels=64, kernel_size=5, strides=2):
        super(ConvDecoder, self).__init__()

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides
        
        # Convolutional transpose layers
        self.conv1 = nn.ConvTranspose2d(self.num_channels, self.num_channels,
                                        kernel_size=self.kernel_size,
                                        stride=2,
                                        padding=self.kernel_size//2,
                                        output_padding=1,
                                        bias=False)
        self.conv2 = nn.ConvTranspose2d(self.num_channels, self.num_channels,
                                        kernel_size=self.kernel_size,
                                        stride=2,
                                        padding=self.kernel_size//2,
                                        output_padding=1,
                                        bias=False)
        self.conv3 = nn.ConvTranspose2d(self.num_channels, self.num_channels,
                                        kernel_size=self.kernel_size,
                                        stride=2,
                                        padding=self.kernel_size//2,
                                        output_padding=1,
                                        bias=False)
        
        # Output convolution to match the input channels (1 in this case)
        self.conv4 = nn.Conv2d(self.num_channels, 1,
                               kernel_size=self.kernel_size,
                               padding=self.kernel_size//2)
        
        # Normalization layers
        self.norm1 = nn.GroupNorm(num_groups=16, num_channels=self.num_channels)
        self.norm2 = nn.GroupNorm(num_groups=16, num_channels=self.num_channels)
        self.norm3 = nn.GroupNorm(num_groups=16, num_channels=self.num_channels)

        # Activation layers
        self.h1 = nn.ReLU()
        self.h2 = nn.ReLU()
        self.h3 = nn.ReLU()

    def forward(self, x):
        # First convolutional block

        print(x.shape)
        x = self.h1(self.norm1(self.conv1(x)))
        print(x.shape)
        # Second convolutional block
        x = self.h2(self.norm2(self.conv2(x)))
        print(x.shape)

        # Third convolutional block
        x = self.h3(self.norm3(self.conv3(x)))
        print(x.shape)

        # Output convolution
        x = self.conv4(x)
        print(x.shape)

        return x


