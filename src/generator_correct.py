import numpy as np
import os
import torch
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    """
    Data Generator for loading batches of data.
    """

    def __init__(self, list_IDs, batch_size, path, scale, input_dim=(1, 400, 400), shuffle=True):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.list_IDs = list_IDs
        self.path = path 
        self.shuffle = shuffle
        self.on_epoch_end()

        # Load scaling factors
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
        # Generate one batch of data.
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
                if not os.path.isfile(file_path):
                    print(f'File {file_path} does not exist.')
                    continue

                # Store sample
                file = np.load(file_path)

                X[i,] = (file - self.x_min) / (self.x_max - self.x_min)
                y[i,] = (file - self.x_min) / (self.x_max - self.x_min)

            except Exception as e:
                print(f'Error loading file {list_ID}: {e}')
                continue

        return X, y
