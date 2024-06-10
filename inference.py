import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from src.blocks import Autoencoder, ConvDecoder, ConvEncoder
from scipy.ndimage import gaussian_filter


def load_model(weights_path, encoder, decoder, device='cpu'):
    """
    Function to load a pre-trained model.

    Parameters:
        weights_path (string): Path to the model weights.
        encoder (torch.nn.Module): The encoder model.
        decoder (torch.nn.Module): The decoder model.
        device (string): The device to load the model on.

    Returns:
        model (torch.nn.Module): The loaded model.
    """
    model = Autoencoder(encoder, decoder)
    map_location = torch.device(device) if torch.cuda.is_available() and 'cuda' in device else torch.device('cpu')
    model.load_state_dict(torch.load(weights_path, map_location=map_location))
    model.to(map_location)
    model.eval()
    return model


def preprocess_image(image_array):
    """
    Function to preprocess an image array.

    Parameters:
        image_array (numpy.ndarray): The image array to preprocess.

    Returns:
        image_tensor (torch.Tensor): The preprocessed image tensor.
    """
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image_array)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor


def get_prediction(model, image_tensor, device):
    """
    Function to get a prediction from a model.

    Parameters:
        model (torch.nn.Module): The model to use for prediction.
        image_tensor (torch.Tensor): The input image tensor.
        device (string): The device to use for prediction.

    Returns:
        output (numpy.ndarray): The model's prediction.
    """
    model.to(device)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return output.cpu().numpy()


def calculate_mse(img1, img2):
    """Helper function to calculate MSE between two images
    
    Parameters:
        img1 (numpy.ndarray): The first image (usually the original image).
        img2 (numpy.ndarray): The second image (usually the output image).
    """
    mse = np.mean((img1 - img2) ** 2)
    return mse


def calculate_similarity(diff, max_diff=1.0):
    """Helper function to calculate similarity between two images
    
    Parameters:
        diff (float): The difference between two images.
        max_diff (float): The maximum possible difference.

    Returns:
        similarity (float): The similarity between the two images as a percentage.
    """
    similarity = (1 - diff / max_diff) * 100
    return similarity


def calculate_psnr(img1, img2, max_pixel_value=1.0):
    """Helper function to calculate PSNR between two images

    Parameters:
        img1 (numpy.ndarray): The first image (usually the original image).
        img2 (numpy.ndarray): The second image (usually the output image).
        max_pixel_value (float): The maximum possible pixel value of the images (default is 255 for 8-bit images).

    Returns:
        float: The Peak Signal-to-Noise Ratio between the two images.
    """
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2, max_pixel_value=1.0, k1=0.01, k2=0.03):
    """Helper function to calculate SSIM between two images

    Parameters:
        img1 (numpy.ndarray): The first image (usually the original image).
        img2 (numpy.ndarray): The second image (usually the output image).
        max_pixel_value (float): The maximum possible pixel value of the images (default is 255 for 8-bit images).
        k1 (float): Constant to stabilize the division with weak denominator (default is 0.01).
        k2 (float): Constant to stabilize the division with weak denominator (default is 0.03).

    Returns:
        float: The Structural Similarity Index between the two images.
    """
    c1 = (k1 * max_pixel_value) ** 2
    c2 = (k2 * max_pixel_value) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = gaussian_filter(img1, sigma=1.5)
    mu2 = gaussian_filter(img2, sigma=1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 ** 2, sigma=1.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def main():
    folder = "data_2d"
    weights = "best_model.pth"
    encoder = ConvEncoder(num_channels=64, kernel_size=5, strides=1, pooling=2)
    decoder = ConvDecoder(num_channels=64, kernel_size=5, strides=2)
    model = load_model(weights, encoder, decoder)

    data_df = pd.read_csv(os.path.join(folder, "file_info.csv"))
    data_df = data_df[data_df["Max_Value"] > 300]
    diff_list, similarity_list, mse_list, similarity_mse_list, psnr_list, ssim_list = [], [], [], [], [], []

    for file_name in data_df["File_Name"]:
        image = np.load(os.path.join(folder, file_name))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalization
        image_tensor = preprocess_image(image)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prediction = get_prediction(model, image_tensor, device)

        mse = calculate_mse(image, prediction)
        diff = image - prediction
        similarity = calculate_similarity(diff)
        similarity_mse = calculate_similarity(mse)
        psnr = calculate_psnr(image, prediction)
        ssim = calculate_ssim(image, prediction)

        diff_list.append(diff)
        similarity_list.append(similarity)
        mse_list.append(mse)
        similarity_mse_list.append(similarity_mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print("Evaluation Results:")
    print(f"Mean Difference: {np.mean(diff_list)}")
    print(f"Mean Similarity: {np.mean(similarity_list)}%")
    print(f"Mean MSE: {np.mean(mse_list)}")
    print(f"Mean MSE Similarity: {np.mean(similarity_mse_list)}%")
    print(f"Mean PSNR: {np.mean(psnr_list)}")
    print(f"Mean SSIM: {np.mean(ssim_list)}")

    model_results = pd.DataFrame({"File_Name": data_df["File_Name"],
                                  "Difference": diff_list,
                                  "Similarity": similarity_list,
                                  "MSE": mse_list,
                                  "Similarity_MSE": similarity_mse_list,
                                  "PSNR": psnr_list,
                                  "SSIM": ssim_list
                                  })
    model_results.to_csv('model_results.csv', index=False)


if __name__ == "__main__":
    main()
