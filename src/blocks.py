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


class Encoder(nn.Module):
    """A simple encoder without any normalization block"""

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 data_size: int,
                 act_fn: object = nn.GELU):
        """
        Initialize an Encoder object.

        :param num_input_channels: Number of input channels
        :param base_channel_size: Number of channels in the first layer
        :param latent_dim: Number of channels in the last layer
        :param data_size: Size of the input data
        :param act_fn: Activation function to use
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Flatten(),
        )
        self.reduced_size = self.compute_output_size(data_size)
        self.linear = nn.Linear(2 * self.reduced_size ** 2 * c_hid, latent_dim)

    def compute_output_size(self, input_size):
        output_size = input_size
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                output_size = ((output_size - layer.kernel_size[0] + 2 * layer.padding[0]) // layer.stride[0]) + 1

        return output_size

    def forward(self, x):
        x = self.net(x)
        return self.linear(x)


class Decoder(nn.Module):
    """A simple encoder without any normalization block"""

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 data_size: int,
                 act_fn: object = nn.GELU):
        """
        Initialize a Decoder object.

        :param num_input_channels: Number of input channels
        :param base_channel_size: Number of channels in the first layer
        :param latent_dim: Number of channels in the last layer
        :param data_size: Size of the input data
        :param act_fn: Activation function to use
        """
        super().__init__()
        c_hid = base_channel_size
        self.data_size = data_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * data_size ** 2 * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 32
            nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, self.data_size, self.data_size)
        x = self.net(x)
        return x


class Autoencoder(nn.Module):
    """A simple autoencoder"""

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 data_size: int = 32,
                 act_fn: object = nn.GELU):
        """
        Initialize an Autoencoder object.

        :param num_input_channels: Number of input channels
        :param base_channel_size: Number of channels in the first layer
        :param latent_dim: Number of channels in the last layer
        :param data_size: Size of the input data
        :param act_fn: Activation function to use
        """
        super().__init__()
        self.encoder = Encoder(num_input_channels, base_channel_size, latent_dim, data_size, act_fn)
        reduced_size = self.encoder.reduced_size
        self.decoder = Decoder(num_input_channels, base_channel_size, latent_dim, reduced_size, act_fn)

    def forward(self, x):
        latent = self.encoder.forward(x)
        x_hat = self.decoder.forward(latent)
        return x_hat


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


    def forward(self, tokens, energy_token):
        # First convolutional block
        x = self.conv1(tokens)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.pool(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.pool(x)

        # Third convolutional block
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.pool(x)

        # Final convolutional block
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.activation(x)

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
                                        stride=self.strides,
                                        padding=self.kernel_size//2,
                                        bias=False)
        self.conv2 = nn.ConvTranspose2d(self.num_channels, self.num_channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.strides,
                                        padding=self.kernel_size//2,
                                        bias=False)
        self.conv3 = nn.ConvTranspose2d(self.num_channels, self.num_channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.strides,
                                        padding=self.kernel_size//2,
                                        bias=False)
        
        # Output convolution
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
        x = self.h1(self.norm1(self.conv1(x)))
        
        # Second convolutional block
        x = self.h2(self.norm2(self.conv2(x)))

        # Third convolutional block
        x = self.h3(self.norm3(self.conv3(x)))

        # Output convolution
        x = self.conv4(x)
        
        return x
