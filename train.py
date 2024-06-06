import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.generator_correct import DataGenerator
from src.blocks import Autoencoder, ConvDecoder, ConvEncoder

batch_size = 20
num_epochs = 1

path = r"data_2d"
data_df = pd.read_csv(os.path.join(path, "file_info.csv"))

data_df = data_df[data_df["Max_Value"] > 300]

scale = {'x_min': data_df["Min_Value"].min(), "x_max": data_df["Max_Value"].max()}

train_split = 0.7
val_split = 0.15
test_split = 0.15

listIDs = data_df['File_Name'].tolist()

random.seed(333)
random.shuffle(listIDs)

trainIDs = listIDs[:int(round(train_split * len(listIDs)))]
valIDs = listIDs[int(round(train_split * len(listIDs))):int(round((train_split + val_split) * len(listIDs)))]
testIDs = listIDs[int(round((train_split + val_split) * len(listIDs))):]

test_df = data_df[data_df['File_Name'].isin(testIDs)]

train_gen = DataGenerator(trainIDs, batch_size, path, scale)
val_gen = DataGenerator(valIDs, batch_size, path, scale)

encoder = ConvEncoder(num_channels=64, kernel_size=5, strides=1, pooling=2)
decoder = ConvDecoder(num_channels=64, kernel_size=5, strides=2)
model = Autoencoder(encoder, decoder)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU idx 2
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_val_loss = float('inf')
best_model_path = 'best_model.pth'

print(f'Number of training batches: {len(train_gen)}')
print(f'Number of validation batches: {len(val_gen)}')

for epoch in range(num_epochs):
    print(f'Starting epoch {epoch+1}')
    model.train()
    train_loss = 0

    # Ensure training loop processes only the expected number of batches
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_gen), total=len(train_gen)):
        try:
            inputs, targets = inputs.to(device), targets.to(device)  # Move to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            print(f'Batch train loss: {loss.item():.4f}')  # Print batch training loss for debugging

        except Exception as e:
            print(f'Error processing batch {batch_idx+1}: {e}')
            continue

    train_loss /= len(train_gen.dataset)
    print(f'Epoch {epoch+1} training completed. Train Loss: {train_loss:.4f}')

    print(f'Validation loop starting for epoch {epoch+1}')
    model.eval()
    val_loss = 0
    batch_count = 0  # Add a counter to count the number of batches
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_gen), total=len(val_gen)):
            try:
                batch_count += 1
                print(f'Processing validation batch {batch_idx+1}/{len(val_gen)}')
                inputs, targets = inputs.to(device), targets.to(device)  # Move to GPU
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                print(f'Batch val loss: {loss.item():.4f}')  # Print batch validation loss for debugging

            except Exception as e:
                print(f'Error processing validation batch {batch_idx+1}: {e}')
                continue

    if batch_count == 0:
        print("No validation batches processed.")

    val_loss /= len(val_gen.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save the model if the validation loss is the lowest we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Model saved with validation loss of {val_loss:.4f}')
