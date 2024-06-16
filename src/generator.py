import numpy as np
import os
import torch
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    """
    Data Generator for loading batches of data.

    args:
        list_IDs (list): List of all data IDs.
        batch_size (int): Size of each batch.
        path (str): Path to the directory containing the data files.
        scale (dict): Dictionary containing the minimum and maximum Hounsfield units, used to scale the data between 0 and 1.
        input_dim (tuple): Dimensions of the input data, standard is (1, 400, 400).
        shuffle (bool): Whether to shuffle the data at the end of each epoch.
       
    """


    def __init__(self, list_IDs, batch_size, path, scale, input_dim=(1, 400, 400), shuffle=True):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.list_IDs = list_IDs
        self.path = path 
        self.shuffle = shuffle
        self.on_epoch_end()

        # Load scaling factors so we normalize the input data
        self.x_min = scale['x_min']
        self.x_max = scale['x_max']
        
    def on_epoch_end(self):
        # Updates indexes after each epoch.
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # Calculates the number of batches per epoch.
       return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch.
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs.
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data.
        X, y = self.__data_generation(list_IDs_temp)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples.
        X = np.empty((self.batch_size, *self.input_dim))
        y = np.empty((self.batch_size, *self.input_dim))
        
        for i, list_ID in enumerate(list_IDs_temp):
            try:
                file_path = os.path.join(self.path, list_ID)
                num_rot = np.random.randint(0, 3) #We randomly rotate the image by 0, 90, 180 or 270 degrees this allows us to augment the data and have 4x the training examples.

                # load file
                file = np.load(file_path)
                file = np.rot90(file, num_rot)

                X[i,] = (file - self.x_min) / (self.x_max - self.x_min)  #Normalize the input data
                y[i,] = (file - self.x_min) / (self.x_max - self.x_min)  #Normalize the target data 

            except Exception as e:
                print(f'Error loading file {list_ID}: {e}')
                continue

        return X, y
