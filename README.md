# Project 6: Reduce Medical Data Dimensionality with Deep Learning
---
This project focuses on reducing the dimensionality of medical CT images using deep learning techniques. By employing an autoencoder architecture, we aim to compress these high-dimensional datasets into a smaller latent space, yielding faster and more efficient processing without significant loss of critical information

## Authors
- Hamdi Elsayed
- Lennard Duynkerke
- Oscar van Baar
- Sergi van den Berg


## Description

Utilizing an Autoencoder architecture, comprising a Convolutional Encoder and Decoder, the project reduces the dimensionality of these images while aiming to retain critical visual information.

### Workflow

1. Data Preprocessing: Transform the 3D CT image data into tensors suitable for model input. The medical scans are loaded and resampled to a uniform voxel spacing to ensure consistenciy in image scale and detail. Then the scans are resized to a standard 400x400x (x,y) dimensions, using padding or croping as necessary. These (still) 3D resized scans are split into 2D slices and saved individually.

2. Model Training: Use the Autoencoder architecture to train the model on a dataset of CT images, optimizing for minimal reconstruction error.

3. Inference and Evaluation: Apply the trained model to new CT images to assess the quality of reconstruction, quantified by metrics such as MSE, PSNR, SSIM.

4. Result Visualization: Generate comparative visualizations between original and reconstructed images alongside performance metrics to validate model effectiveness.

Each step is designed to be modular, allowing for adjustments and improvements such as different model architectures or loss functions (e.g., perceptual loss using VGG for enhanced visual fidelity). In the Usage section we provide a step-by-step guide on model setup, training and evaluation.


## Contents

- *data:* Folder containing preprocessed 2D slices to be used to train the model.
- *results:* Folder containing plots and images that visualize the accuracy and performance of the model.
- *src:* Folder containing source code of the model.
  - blocks.py: blocks like encoder, decoder, and autoencoder used in the model.
  - generator.py: file that defines the datagenerator
- *training:* Folder containing files related to the training of the model.
  - *train.py:* Script that trains the model using MSE loss. Splits data into training, validation, and testing sets and saves the training and validation losses for performance analysis
  - *train_perceptual_loss.py:*  Same as train.py, but now a pre-trained VGG16 model is used to implement perceptual loss during the training of the autoenconder
- *weights:* The weights of the trained model are saved here 
- *preprocess.py:* File used to transform raw data into 2D images that can be used as input data for the model.
- *inference.py:* Script that uses the trained model to encode and decode the test CT images and peforms various performance assessments (MSE, PSNR, and SSIM). It also included functions to load and preprocess image data, execute inference and visualize the outcomes through scatter plots and comparison images

## Usage

The train.py / train_perceptual_loss.py scripts facilitate the training and validation of a deep learning model (using perceptual loss). This script trains the model on images from the 'data' directory, and data splits the data into training, validation, and test sets. During training, the script dynamically adjusts learning rates and continuously evaluates the model against the validation set, saving the model weights to 'Weights/' whenever an improvement in validation loss is observed.

Upon completion of the training phase, the model can be used in inference.py to perform inference on the 'test' subset of the data. This involves encoding and decoding images using the trained autoencoder to assess the quality of image reconstruction. The script also generates performance metrics such as PSNR and SSIM, which are saved for further analysis. Additionally, various performance assessment figures are generated and saved, providing visual and quantitative insights into the model's effectiveness.

## Individual Contribution
- Hamdi: model, train.py, finding right configurations
- Lennard: preprocess.py and majority of inference.py, analyzing data
- Oscar: creating figures, presentation, analyzing data
- Sergi: train_perceptual_loss.py, analyzing data, presentation