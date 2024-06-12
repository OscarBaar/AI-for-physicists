import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import vgg16
from torch.nn import functional as F
from tqdm import tqdm
import pickle
from src.generator_correct import DataGenerator
from src.blocks import Autoencoder, ConvDecoder, ConvEncoder
from torch.optim.lr_scheduler import StepLR

# Load pre-trained VGG16
from torchvision.models import vgg16, VGG16_Weights
vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

# Freeze all VGG parameters
for param in vgg.parameters():
    param.requires_grad = False

# Define perceptual loss
def perceptual_loss(output, target):
    if output.size(1) == 1:
        output = output.repeat(1, 3, 1, 1)
    if target.size(1) == 1:
        target = target.repeat(1, 3, 1, 1)

    resize = transforms.Resize((224, 224))
    output_resized = resize(output)
    target_resized = resize(target)

    output_features = vgg(output_resized)
    target_features = vgg(target_resized)

    return F.mse_loss(output_features, target_features)


path = r"data_2d"
data_df = pd.read_csv(os.path.join(path, "file_info.csv"))
data_df = data_df[data_df["Max_Value"] > 300]
scale = {'x_min': data_df["Min_Value"].min(), "x_max": data_df["Max_Value"].max()}
listIDs = data_df['File_Name'].tolist()

random.seed(333)
random.shuffle(listIDs)

train_split = 0.7
val_split = 0.15
test_split = 0.15
trainIDs = listIDs[:int(round(train_split * len(listIDs)))]
valIDs = listIDs[int(round(train_split * len(listIDs))):int(round((train_split + val_split) * len(listIDs)))]
testIDs = listIDs[int(round((train_split + val_split) * len(listIDs))):]

train_gen = DataGenerator(trainIDs, 10, path, scale)
val_gen = DataGenerator(valIDs, 10, path, scale)

encoder = ConvEncoder(num_channels=64, kernel_size=5, strides=1, pooling=2)
decoder = ConvDecoder(num_channels=64, kernel_size=5, strides=2)
model = Autoencoder(encoder, decoder)
optimizer = optim.Adam(model.parameters(), lr=5e-3)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
vgg.to(device)

best_val_loss = float('inf')
best_model_path = 'best_model.pth'
train_loss_list = []
val_loss_list = []
for epoch in tqdm(range(50)):
    model.train()
    train_loss = 0
    total_train_samples = 0
    for batch_idx,batch in enumerate(train_gen):
        if batch_idx >= len(train_gen):
            break 
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = perceptual_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        total_train_samples += inputs.size(0)

    train_loss /= total_train_samples
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}')
    train_loss_list.append(train_loss)

    model.eval()
    val_loss = 0
    total_val_samples = 0 #Keep track of total samples processed
    with torch.no_grad():
         for batch_idx, batch in enumerate(val_gen):
            if batch_idx >= len(val_gen):
                break
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = perceptual_loss(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            total_val_samples += inputs.size(0)  

     # Save the model if the validation loss is the lowest we've seen so far.
    val_loss /= total_val_samples
    print(f'Epoch {epoch+1}, Val Loss: {val_loss:.6f}')
    val_loss_list.append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print('Model saved')

    scheduler.step()

with open("train_loss.pkl", "wb") as fp:
    pickle.dump(train_loss_list, fp)
with open("val_loss.pkl", "wb") as fp:
    pickle.dump(val_loss_list, fp)
