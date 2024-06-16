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

# Usage



# Contents

- **data:** folder containing preprocessed 2D slices to be used to train the model.
- **results:** folder containing plots and images that visualize the accuracy and performance of the model.
- **src:** folder containing source code of the model.
  - *blocks.py:* blocks like encoder, decoder, and autoencoder used in the model.
  - *generator.py:* Contains the data generator which loads the data into the model.
  - *model.py:* ...
- **training:** folder containing files related to the training of the model. 
  - *train.py:* ...
  - *train_perceptual_loss.py:* ...
  - *best_model.pth:* ...
  - *train_loss.pkl:* ...
  - *val_loss.pkl:* ...
- **preprocess.py:** file used to transform raw data into 2D images that can be used as input data for the model.
- **inference.py:**


# Individual Contribution
- Hamdi: model, training, finding right configurations
- Lennard: preprocess.py and majority of inference.py, analyzing data
- Oscar/Sergi: creating figures, presentation, analyzing data