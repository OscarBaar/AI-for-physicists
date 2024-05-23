import torch.nn as nn


class Encoder(nn.Module):
    """A simple encoder without any normalization block"""
    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Initialize an Encoder object.

        :param num_input_channels: Number of input channels
        :param base_channel_size: Number of channels in the first layer
        :param latent_dim: Number of channels in the last layer
        :param act_fn: Activation function to use

        """
        super().__init__()
        self.size = None
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Flatten(),
        )
        self.linear = nn.Linear(2 * size ** 2 * c_hid, latent_dim)

    def compute_output_size(self, input_size):
        output_size = input_size
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                output_size = ((output_size - layer.kernel_size[0] + 2 * layer.padding[0]) // layer.stride[0]) + 1

        return output_size

    def forward(self, x):
        net = self.net(x)
        self.size = self.compute_output_size(x.shape[0])
        lin = nn.Linear(net, 2 * self.size ** 2 * self.c_hid, self.latent_dim)

        return lin


class Autoencoder(nn.Module):
    """A simple autoencoder"""
    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Initialize an Autoencoder object.

        :param num_input_channels: Number of input channels
        :param base_channel_size: Number of channels in the first layer
        :param latent_dim: Number of channels in the last layer
        :param act_fn: Activation function to use

        """
        super().__init__()
        self.encoder = Encoder(num_input_channels, base_channel_size, latent_dim, act_fn)
        self.decoder = Decoder(num_input_channels, base_channel_size, latent_dim, act_fn)

    def forward(self, x):
        latent = self.encoder.forward(x)
        x_hat = self.decoder.forward(latent)

        return x_hat