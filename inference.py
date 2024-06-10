import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from src.blocks import Autoencoder, ConvDecoder, ConvEncoder


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


    return np.mean((img1 - img2) ** 2)


def calculate_similarity(mse, max_mse):
    """Helper function to calculate similarity between two images
    
    Parameters:
        mse (float): The mean squared error between two images.
        max_mse (float): The maximum possible value of the MSE.

    Returns:
        similarity (float): The similarity between the two images as a percentage.
    """


    similarity = (1 - mse / max_mse) * 100
    return similarity


def evaluate_images(original_folder, reconstructed_folder, max_mse):
    """
    Function to evaluate the accuracy of all reconstructed images with their originals.

    Parameters:
        original_folder (string): Path to the folder containing the original data.
        reconstructed_folder (string): Path to the folder containing the reconstructed data.
        max_mse (float): The maximum value the MSE can take on.

    Returns:
        mse_list (list): Total MSE values for the reconstructed images.
        similarity_list (list): Total similarity values for the reconstructed images.
    """
    original_files = sorted([f for f in os.listdir(original_folder) if f.endswith('.npy')])
    reconstructed_files = sorted([f for f in os.listdir(reconstructed_folder) if f.endswith('.npy')])

    mse_list = []
    similarity_list = []

    for orig_file, recon_file in zip(original_files, reconstructed_files):
        orig_path = os.path.join(original_folder, orig_file)
        recon_path = os.path.join(reconstructed_folder, recon_file)

        orig_image = np.load(orig_path)
        recon_image = np.load(recon_path)

        mse = calculate_mse(orig_image, recon_image)
        similarity = calculate_similarity(mse, max_mse)

        mse_list.append(mse)
        similarity_list.append(similarity)

    return mse_list, similarity_list


def main():
    weights = "best_model.pth"
    encoder = ConvEncoder(num_channels=64, kernel_size=5, strides=1, pooling=2)
    decoder = ConvDecoder(num_channels=64, kernel_size=5, strides=2)
    model = load_model(weights, encoder, decoder)
    max_mse = 1024 ** 2
    folder = "data_2d"

    # mse_list, similarity_list = evaluate_images("data_2d", reconstructed_folder, max_mse)

    file_names = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    mse_list, similarity_list = [], []
    for file_name in file_names:
        image_array = np.load(os.path.join(folder, file_name))  # Load your image array here
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        image_tensor = preprocess_image(image_array)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prediction = get_prediction(model, image_tensor, device)

        mse = calculate_mse(image_array, prediction)
        similarity = calculate_similarity(mse, max_mse)
        mse_list.append(mse)
        similarity_list.append(similarity)

    print("Evaluation Results:")
    print(f"Mean MSE: {np.mean(mse_list)}")
    print(f"Mean Similarity: {np.mean(similarity_list)}%")
    print(f"Individual Similarity values: {similarity_list}")

    model_results = pd.DataFrame({"file": file_names, "mse": mse_list, "similarity": similarity_list})
    model_results.to_csv('model_results.csv', index=False)


if __name__ == "__main__":
    main()
