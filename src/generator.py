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