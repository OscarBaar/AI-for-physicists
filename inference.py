import os
import numpy as np


def calculate_mse(img1, img2):
    """Helper function to calculate MSE between two images"""

    return np.mean((img1 - img2) ** 2)


def calculate_similarity(mse, max_mse):
    """Helper function to calculate similarity between two images"""
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
    original_folder = 'original_2d'
    reconstructed_folder = 'reconstructed_2d'
    max_mse = 1024 ** 2

    mse_list, similarity_list = evaluate_images(original_folder, reconstructed_folder, max_mse)

    print("Evaluation Results:")
    print(f"Mean MSE: {np.mean(mse_list)}")
    print(f"Mean Similarity: {np.mean(similarity_list)}%")
    print(f"Individual Similarity values: {similarity_list}")


if __name__ == "__main__":
    main()
