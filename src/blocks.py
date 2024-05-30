import warnings # ignore future warnings for now
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path # Path library

import numpy as np # numpy library
import pandas as pd # pandas library
import seaborn as sns # seaborn library for plotting figures
import random

import torch # pytorch library
import torch.nn as nn # neural network module
import torch.optim as optim # optimization module
from torch.utils.data import DataLoader # data loader module
import torchvision.transforms as transforms # image transforms module

from matplotlib import pyplot as plt # plotting library
from IPython.core.display import SVG # display SVG images in Jupyter
from tqdm import tqdm, trange # progress bar library


# set seaborn style to no grid and white background
sns.set(style='white', rc={'axes.grid': False})


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
