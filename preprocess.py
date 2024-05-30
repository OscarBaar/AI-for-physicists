import numpy as np
import pandas as pd
import os


def resize_scan(scan, target_size=400, pad_value=-1024):
    """
    Resize a 3D numpy array in the x and y dimensions to a fixed size.
    Smaller arrays are padded with a specified value, larger arrays are cropped.

    Parameters:
    scan (np.ndarray): The 3D input array to be resized.
    target_size (int): The target size for the x and y dimensions.
    pad_value (int): The value to use for padding.

    Returns:
    new_scan (np.ndarray): The resized 3D array.
    """
    z, y, x = scan.shape
    new_scan = np.full((z, target_size, target_size), pad_value, dtype=scan.dtype)

    y_center, x_center = y // 2, x // 2
    y_start = max(0, y_center - target_size // 2)
    x_start = max(0, x_center - target_size // 2)
    y_end = min(y, y_center + target_size // 2)
    x_end = min(x, x_center + target_size // 2)

    new_y_start = max(0, target_size // 2 - y_center)
    new_x_start = max(0, target_size // 2 - x_center)
    new_y_end = new_y_start + (y_end - y_start)
    new_x_end = new_x_start + (x_end - x_start)

    new_scan[:, new_y_start:new_y_end, new_x_start:new_x_end] = scan[:, y_start:y_end, x_start:x_end]

    return new_scan


def split_scan(scan, file_name, output_dir='data_2d/'):
    """
    Split a resized 3D numpy array into individual 2D images of size 400x400,
    save each image as a separate file, and create a DataFrame containing information
    about each image, including the original file name, the image number, and the
    minimum, maximum, and mean values of the image.

    Parameters:
    resized_array (numpy.ndarray): Resized 3D numpy array.
    output_dir (str): Path to the directory where the split images will be saved.

    Returns:
    file_info (pd.DataFrame): DataFrame containing information about each image.
    """
    file_idx = int(file_name.split("_")[-1].split(".")[0])

    # Initialize lists to store data
    slice_names = []
    file_names = []
    file_indices = []
    image_numbers = []
    min_values = []
    max_values = []
    mean_values = []

    for i, slice_2d in enumerate(scan):
        slice_file_name = f'img_{file_idx}_slice_{i}.npy'

        np.save(os.path.join(output_dir, slice_file_name), slice_2d)

        # Store information about the slice
        slice_names.append(slice_file_name)
        file_names.append(file_name)
        file_indices.append(file_idx)
        image_numbers.append(i)
        min_values.append(np.min(slice_2d))
        max_values.append(np.max(slice_2d))
        mean_values.append(np.mean(slice_2d))

    data = {
        'File_Name': slice_names,
        'Origin_File': file_names,
        'Origin_Index': file_indices,
        'Image_Number': image_numbers,
        'Min_Value': min_values,
        'Max_Value': max_values,
        'Mean_Value': mean_values
    }

    file_info = pd.DataFrame(data)

    return file_info


if __name__ == "__main__":
    folder = "lung_cts/"
    dfs = []
    for file in os.listdir(folder):
        if not file.endswith(".npy"):
            continue

        scan = np.load(folder + file)
        resized_scan = resize_scan(scan)
        df = split_scan(resized_scan, file)
        dfs.append(df)

        print(file, scan.shape)

        del scan, resized_scan

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv('data_2d/file_info.csv', index=False)
