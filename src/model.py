import numpy as np
import torch
from src.blocks import ConvEncoder, ConvDecoder



def autoencoder(input_shape=(1,400,400),kernel_size=5):
    
    input_shape = input_shape
    kernel_size = kernel_size


    inputs= torch.nn.Input(input_shape)

    latent_space= ConvEncoder(inputs, kernel_size=kernel_size)

    outputs = ConvDecoder(latent_space, kernel_size=kernel_size)


    model = torch.nn.Model(inputs, outputs)
    return model

