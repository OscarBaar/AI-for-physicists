import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from src.blocks import Autoencoder, ConvDecoder, ConvEncoder
from scipy.ndimage import gaussian_filter
import time
import random
import matplotlib.pyplot as plt
import json


def load_model(weights_path, encoder, decoder, device='cpu'):
    """
    Function to load a pre-trained old_model.

    Parameters:
        weights_path (string): Path to the old_model weights.
        encoder (torch.nn.Module): The encoder old_model.
        decoder (torch.nn.Module): The decoder old_model.
        device (string): The device to load the old_model on.

    Returns:
        old_model (torch.nn.Module): The loaded old_model.
    """
    model = Autoencoder(encoder, decoder)

    if torch.cuda.is_available() and 'cuda' in device:
        model = model.cuda()
        map_location = torch.device(device)
    else:
        map_location = torch.device('cpu')

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
        model (torch.nn.Module): The old_model to use for prediction.
        image_tensor (torch.Tensor): The input image tensor.
        device (string): The device to use for prediction.

    Returns:
        output (numpy.ndarray): The old_model's prediction.
        encode_time (float): Time taken to encode the image.
        decode_time (float): Time taken to decode the image.
    """
    encoder, decoder = model.encoder, model.decoder 
    image_tensor = image_tensor.to(device)
 
    start_encode = time.time()
    with torch.no_grad():
        encoded = encoder(image_tensor)
    encode_time = time.time() - start_encode

    start_decode = time.time()
    with torch.no_grad():
        output = decoder(encoded)
    decode_time = time.time() - start_decode

    return output.detach().cpu().numpy(), encode_time, decode_time


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


def infer_all(model, folder, df, device, output_dir):
    """
    Apply the old_model to all images from df,

    Parameters:
        model (torch.old_model): The autoencoder old_model to be used for inference.
        folder (string): Folder containing the original images.
        df (pd.DataFrame): Information about the original images.
        device (string): The device to use for prediction.

    Returns:
        None
    """
    (err_list, similarity_list, mse_list, similarity_mse_list, psnr_list,
     ssim_list, encode_times, decode_times) = [], [], [], [], [], [], [], []

    for file_name in df['File_Name']:
        print(file_name)
        image = np.load(os.path.join(folder, file_name))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalization
        image_tensor = preprocess_image(image)

        # Get prediction
        prediction, encode_time, decode_time = get_prediction(model, image_tensor, device)
        encode_times.append(encode_time)
        decode_times.append(decode_time)

        mse = calculate_mse(image, prediction)
        err = np.mean(np.abs(image - prediction))
        similarity = calculate_similarity(err)
        similarity_mse = calculate_similarity(mse)
        psnr = calculate_psnr(image, prediction)
        ssim = calculate_ssim(image, prediction)

        err_list.append(err)
        similarity_list.append(similarity)
        mse_list.append(mse)
        similarity_mse_list.append(similarity_mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print('Evaluation Results:')
    print(f'Mean error: {np.mean(err_list)}')
    print(f'Mean similarity: {np.mean(similarity_list)}')
    print(f'Mean MSE: {np.mean(mse_list)}')
    print(f'Mean MSE similarity: {np.mean(similarity_mse_list)}')
    print(f'Mean PSNR: {np.mean(psnr_list)}')
    print(f'Mean SSIM: {np.mean(ssim_list)}')
    print(f'Mean encode time: {np.mean(encode_times)} seconds')
    print(f'Mean decode time: {np.mean(decode_times)} seconds')

    # Create DataFrame to store results along with inference times
    model_results = pd.DataFrame({
        'File_Name': df['File_Name'],
        'Error': err_list,
        'Similarity': similarity_list,
        'MSE': mse_list,
        'Similarity_MSE': similarity_mse_list,
        'PSNR': psnr_list,
        'SSIM': ssim_list,
        'Encode_Time': encode_times,
        'Decode_Time': decode_times
    })

    model_results.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)


def plot_images(original_images, predictions, titles, file_info, output_dir):
    """
    Plots best and worst results of the old_model according to different parameters.

    Parameters:
        original_images (list of np.ndarray): List of original images.
        predictions (list of np.ndarray): List of predicted images by the old_model.
        titles (list of str): List of titles for the plots.
        file_info (list of dict): List of dictionaries containing detailed information about each image.

    Returns:
        None
    """
    def extract_file_name(title):
        """Extract file name from title"""
        start = title.find('(') + 1
        end = title.find(')', start)
        if 0 < start < end:
            file_name = title[start:end].strip().replace(' ', '_')
            return file_name + '.jpg'
        return None

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(len(original_images)):
        difference = predictions[i] - original_images[i]
        v_min, v_max = -np.max(np.abs(difference)), np.max(np.abs(difference))
        # Plot original image
        im1 = axes[i, 0].imshow(original_images[i], cmap='gray', aspect='auto')
        axes[i, 0].set_title(f'Original - {titles[i]}')
        axes[i, 0].axis('off')

        # Plot prediction image
        im2 = axes[i, 1].imshow(predictions[i], cmap='gray', aspect='auto')
        axes[i, 1].set_title(f'Prediction - {titles[i]}')
        axes[i, 1].axis('off')

        # Plot difference using heatmap
        im3 = axes[i, 2].imshow(difference, cmap='seismic', vmin=v_min, vmax=v_max, aspect='auto')
        axes[i, 2].set_title(f'Difference - {titles[i]}')
        axes[i, 2].axis('off')
        # Print variables and values aligned
        info = file_info[i]
        max_length = max(len(name) for name in info.keys())
        for name, value in info.items():
            print(f'{name:<{max_length}}: {value}')
        print('\n')

    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Right side colorbar
    cbar = fig.colorbar(im3, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Difference', rotation=270, labelpad=15)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(output_dir, extract_file_name(titles[0])))
    plt.show()


def plot_results(model, folder, file_df, results_df, device, output_dir):
    """
    Plots and stores best and worst results of the model according to different parameters.

    Parameters:
        model (torch.model): The autoencoder model to be used for inference.
        folder (string): Folder containing the original images.
        file_df (pd.DataFrame): Information about the original images.
        results_df (pd.DataFrame): Information about the inference results.
        device (string): The device to use for prediction.

    Returns:
        None
    """
    columns = results_df.columns
    num_images = 2
    results_storage = []

    for column in columns:
        if column == 'File_Name':
            continue

        highest_val = results_df.nlargest(num_images, column)
        lowest_val = results_df.nsmallest(num_images, column)

        for data, label in [(highest_val, 'High'), (lowest_val, 'Low')]:
            images, predictions, titles, file_infos = [], [], [], []
            for idx, row in data.iterrows():
                file_name = row['File_Name']
                image = np.load(os.path.join(folder, file_name))
                image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalization
                image_tensor = preprocess_image(image)
                prediction, _, _ = get_prediction(model, image_tensor, device)

                images.append(image)
                predictions.append(prediction[0, 0, :, :])  # Adjust this index if your prediction array shape is different
                titles.append(f'{file_name} ({label} {column})')

                file_info = {col: row[col] for col in results_df.columns}
                file_info.update(file_df[file_df['File_Name'] == file_name].to_dict('records')[0])
                file_infos.append(file_info)

                result_details = {
                    'type': label,
                    'parameter': column,
                    'file_name': file_name,
                    'image': image,
                    'prediction': prediction,
                    'additional_info': file_info
                }
                results_storage.append(result_details)

            # Plot images and print parameters
            print(f'Plotting for parameter: {column} ({label})')
            plot_images(images, predictions, titles, file_infos, output_dir)
    metadata_path = os.path.join(output_dir, 'results_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(results_storage, f, indent=4, default=str)

def create_scatterplots(results, folder):
    """
    Plots scatterplots of important observables to show correlations.

    Parameters:
        results (pd.DataFrame): Information about the inference results.
        folder (string): Destination folder to save the plots to.

    Returns:
        None
    """
    plt.figure()
    plt.title('PSNR vs. SSIM')
    plt.scatter(results['PSNR'], results['SSIM'])
    plt.xlabel('PSNR')
    plt.ylabel('SSIM')
    plt.savefig(os.path.join(folder, 'PSNRvsSSIM.png'))
    plt.show()

    plt.figure()
    plt.title('MSE vs. PSNR')
    plt.scatter(results['MSE'], results['PSNR'])
    plt.xlabel('MSE')
    plt.ylabel('PSNR')
    plt.savefig(os.path.join(folder, 'MSEvsPSNR.png'))
    plt.show()

    plt.figure()
    plt.title('MSE vs. SSIM')
    plt.scatter(results['MSE'], results['SSIM'])
    plt.xlabel('MSE')
    plt.ylabel('SSIM')
    plt.savefig(os.path.join(folder, 'MSEvsSSIM.png'))
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(results['Mean_Value'], results['SSIM'], label='SSIM')
    ax[0].set_xlabel('Mean Intensity (HU)')
    ax[0].set_ylabel('SSIM')

    ax[1].scatter(results['Mean_Value'], results['PSNR'], label='PSNR')
    ax[1].set_xlabel('Mean Intensity (HU)')
    ax[1].set_ylabel('PSNR')

    ax[2].scatter(results['Mean_Value'], results['MSE'], label='MSE')
    ax[2].set_xlabel('Mean Intensity (HU)')
    ax[2].set_ylabel('MSE')

    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'Intensity.png'))
    fig.show()


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_folder = 'data'
    results_folder = 'results_MSELoss'
    import os

# Define the name of the folder


# Check if the folder exists
    if not os.path.exists(results_folder):
    # If the folder does not exist, create it
        os.makedirs(results_folder)
        print(f"Folder '{results_folder}' created.")
    else:
    # Prompt the user to confirm overwrite
        response = input(f"Folder '{results_folder}' already exists. Do you want to continue and possibly overwrite files? (yes/no): ")
        if response.lower() != 'yes':
            raise Exception("Operation cancelled by user to avoid potential file overwrite.")
        else:
            print("Proceeding with operations that may overwrite files in the existing folder.")

    weights = os.path.join('Weights', 'ModelWeightsMSELoss.pth')
    encoder = ConvEncoder(num_channels=64, kernel_size=5, strides=1, pooling=2)
    decoder = ConvDecoder(num_channels=64, kernel_size=5, strides=2)
    model = load_model(weights, encoder, decoder, device)

    # Load data
    data_df = pd.read_csv(os.path.join(data_folder, 'file_info.csv'))
    data_df = data_df[data_df['Max_Value'] > 300]
    
    # Split data to ensure using only test data
    train_split, val_split, test_split = 0.7, 0.15, 0.15
    list_ids = data_df['File_Name'].tolist()
    random.seed(333)
    random.shuffle(list_ids)

    num_train = int(round(train_split * len(list_ids)))
    num_val = int(round(val_split * len(list_ids)))

    test_ids = list_ids[num_train + num_val:]
    test_df = data_df[data_df['File_Name'].isin(test_ids)]

    # Apply inference to all images
    infer_all(model, data_folder, test_df, device, results_folder)

    # Plot best and worst performing images
    model_results = pd.read_csv(os.path.join(results_folder, 'model_results.csv'))
    model_results = (model_results.merge(data_df, on='File_Name', how='left'))
    plot_results(model, data_folder, test_df, model_results, device, results_folder)
    
    # Scatterplots
    create_scatterplots(model_results, results_folder)


if __name__ == "__main__":
    main()
