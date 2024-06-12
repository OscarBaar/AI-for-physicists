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


def infer_all(model, data_folder, df):
    """
    Apply the model to all images from df,

    Parameters:
        model (torch.model): The autoencoder model to be used for inference.
        data_folder (string): Folder containing the original images.
        df (pd.DataFrame): Information about the original images.

    Returns:
        None
    """
    (err_list, similarity_list, mse_list, similarity_mse_list,
     psnr_list, ssim_list, inference_times) = [], [], [], [], [], [], []

    for file_name in df["File_Name"]:
        print(file_name)
        image = np.load(os.path.join(data_folder, file_name))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalization
        image_tensor = preprocess_image(image)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get prediction
        prediction, encode_time, decode_time = get_prediction(model, image_tensor, device)
        total_inference_time = encode_time + decode_time
        inference_times.append(total_inference_time)

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

    print("Evaluation Results:")
    print(f"Mean error: {np.mean(err_list)}")
    print(f"Mean similarity: {np.mean(similarity_list)}")
    print(f"Mean MSE: {np.mean(mse_list)}")
    print(f"Mean MSE similarity: {np.mean(similarity_mse_list)}")
    print(f"Mean PSNR: {np.mean(psnr_list)}")
    print(f"Mean SSIM: {np.mean(ssim_list)}")
    print(f"Mean inference time: {np.mean(inference_times)} seconds")

    # Create DataFrame to store results along with inference times
    model_results = pd.DataFrame({
        "File_Name": df["File_Name"],
        "Error": err_list,
        "Similarity": similarity_list,
        "MSE": mse_list,
        "Similarity_MSE": similarity_mse_list,
        "PSNR": psnr_list,
        "SSIM": ssim_list,
        "Inference_Time": inference_times
    })

    model_results.to_csv('model_results.csv', index=False)


def plot_images(original_images, predictions, titles, file_info):
    """
    Plots best and worst results of the model according to different parameters.

    Parameters:
        original_images (list of np.ndarray): List of original images.
        predictions (list of np.ndarray): List of predicted images by the model.
        titles (list of str): List of titles for the plots.
        file_info (list of dict): List of dictionaries containing detailed information about each image.

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i in range(len(original_images)):
        axes[i, 0].imshow(original_images[i], cmap='gray')
        axes[i, 0].set_title(f'Original - {titles[i]}')
        axes[i, 1].imshow(predictions[i], cmap='gray')
        axes[i, 1].set_title(f'Prediction - {titles[i]}')

        # Print variables and values aligned
        info = file_info[i]
        max_length = max(len(name) for name in info.keys())
        for name, value in info.items():
            print(f"{name:<{max_length}}: {value}")
        print('\n')

    plt.tight_layout()
    plt.savefig(os.path.join('results', f'{titles[i]}.jpg'))
    plt.show()


def plot_results(model, data_folder, file_df, results_df):
    """
    Plots best and worst results of the model according to different parameters.

    Parameters:
        model (torch.model): The autoencoder model to be used for inference.
        data_folder (string): Folder containing the original images.
        file_df (pd.DataFrame): Information about the original images.
        results_df (pd.DataFrame): Information about the inference results.

    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    columns = results_df.columns
    num_images = 2

    for column in columns:
        if column == 'File_Name':
            continue

        highest_val = results_df.nlargest(num_images, column)
        lowest_val = results_df.nsmallest(num_images, column)

        # Get results of highest values
        images_high, predictions_high, titles_high, file_info_high = [], [], [], []
        for idx, row in highest_val.iterrows():
            file_name = row["File_Name"]
            image = np.load(os.path.join(data_folder, file_name))
            image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalization
            image_tensor = preprocess_image(image)
            prediction, _, _ = get_prediction(model, image_tensor, device)

            images_high.append(image)
            predictions_high.append(prediction[0, 0, :, :])
            titles_high.append(f'{file_name} (High {column})')

            file_info = {col: row[col] for col in results_df.columns}
            file_info.update(file_df[file_df['File_Name'] == file_name].to_dict('records')[0])
            file_info_high.append(file_info)

        # Get results of lowest values
        images_low, predictions_low, titles_low, file_info_low = [], [], [], []
        for idx, row in lowest_val.iterrows():
            file_name = row["File_Name"]
            image = np.load(os.path.join(data_folder, file_name))
            image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalization
            image_tensor = preprocess_image(image)
            prediction, _, _ = get_prediction(model, image_tensor, device)
            np.save(os.path.join('predictions', 'pred_'+file_name), prediction)

            images_low.append(image)
            predictions_low.append(prediction[0, 0, :, :])
            titles_low.append(f'{file_name} (Low {column})')

            file_info = {col: row[col] for col in results_df.columns}
            file_info.update(file_df[file_df['File_Name'] == file_name].to_dict('records')[0])
            file_info_low.append(file_info)

        # Plot images and print parameters
        print(f'Plotting for parameter: {column}')
        plot_images(images_high, predictions_high, titles_high, file_info_high)
        plot_images(images_low, predictions_low, titles_low, file_info_low)


def main():
    folder = "data_2d"
    weights = "best_model.pth"
    encoder = ConvEncoder(num_channels=64, kernel_size=5, strides=1, pooling=2)
    decoder = ConvDecoder(num_channels=64, kernel_size=5, strides=2)
    model = load_model(weights, encoder, decoder)

    # Load data
    data_df = pd.read_csv(os.path.join(folder, "file_info.csv"))
    data_df = data_df[data_df["Max_Value"] > 300]
    
    # Split data to ensure using only test data
    train_split, val_split, test_split = 0.7, 0.15, 0.15
    list_ids = data_df['File_Name'].tolist()
    random.seed(333)
    random.shuffle(list_ids)

    num_train = int(round(train_split * len(list_ids)))
    num_val = int(round(val_split * len(list_ids)))

    test_ids = list_ids[num_train + num_val:]
    test_df = data_df[data_df['File_Name'].isin(test_ids)]
    print(len(test_df))

    # Apply inference to all images
    # infer_all(model, folder, test_df)

    # Plot best and worst performing images
    model_results = pd.read_csv('model_results.csv')
    model_results = (model_results.merge(data_df, on='File_Name', how='left'))
    # plot_results(model, folder, test_df, model_results)

    # Create scatterplots
    plt.figure()
    plt.title("PSNR vs. SSIM")
    plt.scatter(model_results['PSNR'], model_results['SSIM'])
    plt.xlabel("PSNR")
    plt.ylabel("SSIM")
    plt.savefig(os.path.join('results', 'PSNRvsSSIM.png'))
    plt.show()

    plt.figure()
    plt.title("MSE vs. PSNR")
    plt.scatter(model_results['MSE'], model_results['PSNR'])
    plt.xlabel("MSE")
    plt.ylabel("PSNR")
    plt.savefig(os.path.join('results', 'MSEvsPSNR.png'))
    plt.show()

    plt.figure()
    plt.title("MSE vs. SSIM")
    plt.scatter(model_results['MSE'], model_results['SSIM'])
    plt.xlabel("MSE")
    plt.ylabel("SSIM")
    plt.savefig(os.path.join('results', 'MSEvsSSIM.png'))
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(model_results['Mean_Value'], model_results['SSIM'], label="SSIM")
    ax[0].set_xlabel("Mean Intensity (HU)")
    ax[0].set_ylabel("SSIM")

    ax[1].scatter(model_results['Mean_Value'], model_results['PSNR'], label="PSNR")
    ax[1].set_xlabel("Mean Intensity (HU)")
    ax[1].set_ylabel("PSNR")

    ax[2].scatter(model_results['Mean_Value'], model_results['MSE'], label="MSE")
    ax[2].set_xlabel("Mean Intensity (HU)")
    ax[2].set_ylabel("MSE")

    fig.tight_layout()
    fig.savefig("Intensity.png")
    fig.show()


if __name__ == "__main__":
    main()
