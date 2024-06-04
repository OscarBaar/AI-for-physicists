import numpy as np
import torch
from src.blocks import ConvEncoder, ConvDecoder

import torch.nn as nn




def autoencoder(input_shape=(1,400,400),kernel_size=5):
    
    input_shape = input_shape
    kernel_size = kernel_size


    inputs= torch.nn.Input(input_shape)

    latent_space= ConvEncoder(inputs, kernel_size=kernel_size)

    outputs = ConvDecoder(latent_space, kernel_size=kernel_size)


    model = torch.nn.Model(inputs, outputs)
    return model

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed = self.decoder(latent_space)
        return reconstructed