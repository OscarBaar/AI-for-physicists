import torch.nn as nn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Ignore future warnings for now


class ConvEncoder(nn.Module):
    """
    Convolutional Encoder with multiple layers of convolution, pooling, (Batch) normalization, and (Relu) activation.

    The input data is a tensor of shape (batch_size, 1, height, width), where height and width are the dimensions of the input image, in this case 400 x 400.

    The Convolutional Encoder reduces the dimensionality 2**3 times in each dimension, resulting in a tensor of shape (batch_size, num_channels, height//8, width//8).

    The Convoltuional Layers are padded to keep the spatial dimensions the same, and the (Max) pooling layers reduce the spatial dimensions by a factor of 2.


    Args:
        num_channels (int): Number of channels for convolutional layers. Default is 64.
        kernel_size (int): Size of the convolutional kernel. Default is 5.
        strides (int): Stride size for convolutional layers. Default is 1.
        pooling (int): Size of the pooling window and stride. Default is 2.
    """
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
        # Input shape: (batch_size, 1, 400, 400)
        # Output shape: (batch_size, num_channels, 200, 200)
        x = self.conv1(tokens)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.pool(x)
       
        # Second convolutional block
        # Input shape: (batch_size, num_channels, 200, 200)
        # Output shape: (batch_size, num_channels, 100, 100)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.pool(x)
         
        # Third convolutional block
        # Input shape: (batch_size, num_channels, 100, 100)
        # Output shape: (batch_size, num_channels, 50, 50)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # Final convolutional block
        # Input shape: (batch_size, num_channels, 50, 50)
        # Output shape: (batch_size, num_channels, 50, 50)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.activation(x)

        return x


class ConvDecoder(nn.Module):

    """
    Convolutional Decoder with multiple layers of transposed convolution, Bathch normalization, and Relu activation.

    The Transposed Convolution upsamples the downsampled latenst space to the original image dimensions.



    Args:
        num_channels (int): Number of channels for transposed convolutional layers. Default is 64.
        kernel_size (int): Size of the convolutional kernel. Default is 5.
        strides (int): Stride size for transposed convolutional layers. Default is 2.
    """

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
        # Input shape: (batch_size, num_channels, 50, 50) , Output shape: (batch_size, num_channels, 100, 100)
        x = self.h1(self.norm1(self.conv1(x)))

        # Second convolutional block
        # Input shape: (batch_size, num_channels, 100, 100) , Output shape: (batch_size, num_channels, 200, 200)
        x = self.h2(self.norm2(self.conv2(x)))

        # Third convolutional block
        # Input shape: (batch_size, num_channels, 200, 200) , Output shape: (batch_size, num_channels, 400, 400)
        x = self.h3(self.norm3(self.conv3(x)))

        # Output convolution
        # Input shape: (batch_size, num_channels, 400, 400) , Output shape: (batch_size, 1, 400, 400)
        x = self.conv4(x)

        return x


class Autoencoder(nn.Module):

    """
    Autoencoder old_model consisting of an encoder and a decoder.

    Args:
        encoder (nn.Module): Encoder old_model instance.
        decoder (nn.Module): Decoder old_model instance.
    """
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # Input shape: (batch_size, 1, 400, 400), Output shape: (batch_size, num_channels, 50, 50)
        latent_space = self.encoder(x)

        # Input shape: (batch_size, num_channels, 50, 50) , Output shape: (batch_size, 1, 400, 400)
        reconstructed = self.decoder(latent_space)
        return reconstructed
