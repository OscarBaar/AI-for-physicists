import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from src.generator import DataGenerator
from src.blocks import Autoencoder, ConvDecoder, ConvEncoder
from torch.optim.lr_scheduler import StepLR

batch_size = 10
num_epochs = 50

path = r"data"
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
optimizer = optim.Adam(model.parameters(), lr=5e-3)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
# Move the old_model to GPU idx 2
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_val_loss = float('inf')
best_model_path = 'Weights/ModelWeightsMSELoss.pth'

print(f'Number of training batches: {len(train_gen)}')
print(f'Number of validation batches: {len(val_gen)}')
train_loss_list = []
val_loss_list = []
for epoch in range(num_epochs):
    print(f'Starting epoch {epoch+1}')
    model.train()
    train_loss = 0
    total_train_samples = 0  # Keep track of total samples processed

    for batch_idx, batch in enumerate(train_gen):
        if batch_idx >= len(train_gen):
            break 
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        train_loss += batch_loss * inputs.size(0)
        total_train_samples += inputs.size(0)
        #print(f'Batch {batch_idx + 1}/{len(train_gen)}, Train Loss: {batch_loss:.4f}')

    train_loss /= total_train_samples
    print(f'Epoch {epoch+1} training completed. Average Train Loss: {train_loss:.4f}')
    train_loss_list.append(train_loss)
    model.eval()
    val_loss = 0
    total_val_samples = 0  # Keep track of total samples processed

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_gen):
            if batch_idx >= len(val_gen):
                break
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_loss = loss.item()
            val_loss += batch_loss * inputs.size(0)
            total_val_samples += inputs.size(0)
            #print(f'Batch {batch_idx + 1}/{len(val_gen)}, Val Loss: {batch_loss:.4f}')

    val_loss /= total_val_samples
    print(f'Epoch {epoch+1}, Average Val Loss: {val_loss:.4f}')
    val_loss_list.append(val_loss)

    # Save the old_model if the validation loss is the lowest we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Model saved with validation loss of {val_loss:.4f}')
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]  # Get the last learning rate
    print(f'Epoch {epoch+1}/{num_epochs}, Current learning rate: {current_lr}')

with open("training/train_lossMSE", "wb") as fp:  # Saving the training loss
    pickle.dump(train_loss_list, fp)
with open("training/val_lossMSE", "wb") as fp:  #Saving the validation loss
    pickle.dump(val_loss_list, fp)
