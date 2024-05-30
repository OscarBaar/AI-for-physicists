import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def load_img(file_path, noise_mean=0, noise_std=0.01):
    """
    Load a .npy file, apply a random rotation (90, 180, 270 degrees),
    and add random Gaussian noise to the data.

    Parameters:
    file_path (str): Path to the .npy file to be loaded.
    noise_mean (float): Mean of the Gaussian noise to be added. Default is 0.
    noise_std (float): Standard deviation of the Gaussian noise to be added. Default is 0.01.

    Returns:
    tensor_scan (torch.Tensor): The processed image tensor.
    """
    scan = np.load(file_path)

    rotate = random.choice([1, 2, 3])
    scan = np.rot90(scan, k=rotate, axes=(0, 1))
    noise = np.random.normal(noise_mean, noise_std, scan.shape)

    noisy_scan = scan + noise
    tensor_scan = torch.tensor(noisy_scan, dtype=torch.float32).unsqueeze(0)

    return tensor_scan


def plot_2d_slice(data, slice_number):
    """
    Plot a 2D slice of a 3D CT scan.

    Parameters:
    data (numpy.ndarray): The 3D CT scan data.
    slice_number (int): The slice number to be plotted.

    """
    plt.imshow(data[slice_number+1], cmap='gray')
    plt.title(f'Slice {slice_number} of the CT Scan')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    folder = r"lung_cts/"
    for file in os.listdir(folder):
        if not file.endswith(".npy"):
            continue

        scan = np.load(os.path.join(folder,file))
        plot_2d_slice(scan, 50)
        del scan

