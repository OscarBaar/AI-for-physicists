import numpy as np
import torch
from blocks import Autoencoder, ConvEncoder, ConvDecoder
from torchsummary import summary

def autoencoder(input_shape=(1,400,400),kernel_size=5):
    
    input_shape = input_shape
    kernel_size = kernel_size


    inputs= torch.nn.Input(input_shape)

    latent_space= ConvEncoder(inputs, kernel_size=kernel_size)

    outputs = ConvDecoder(latent_space, kernel_size=kernel_size)


    model = torch.nn.Model(inputs, outputs)
    return model


num_channels = 128  # Number of channels in the encoder and decoder
input_image_dimensions = (1, 400, 400)  # Format (Channels, Height, Width)


encoder = ConvEncoder(num_channels=num_channels, kernel_size=5, strides=1, pooling=2)
decoder = ConvDecoder(num_channels=num_channels, kernel_size=5, strides=2)

model = Autoencoder(encoder=encoder, decoder=decoder)

# Use torchsummary to print the model summary
model.to('cuda' if torch.cuda.is_available() else 'cpu')

summary(model, input_size=input_image_dimensions)