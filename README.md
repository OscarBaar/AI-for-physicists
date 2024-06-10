# AI-for-physicists

## To-do's 
For next Friday:
- Initial training results
- Initial inference results
- Summary of problematic cases (what kind of images are difficult - general look, doesn't have to be really in-depth)
- Summary of training + inference times.
 


## Table of Contents

1. blocks.py
Script that defines the building blocks of the model

2. generator_correct.py
Script contaning 'DataGenerator' class, which is used for:
* Loading and batching image data from disk for training and evaluation
* Applying preprocessing steps like normalization or data augmentation (e.g. rotation) directly within data loading workflow

3. train.py
Script handles training of the autoencoder model
* Reads and prepares training and validation datasets
* SEts up model, loss, optimizer and learning rate scheduler
* Executres training loop, periodically validating validating model on a separate datatest to monitor performance
* Saves best performing model

4. inference.py
Deployment of the model
* Performs inference to generate predictions from input images
* Evaluates the model's output by comparing predictions against ground truth data

5. evaluate.py
Evaluating the training model
* Calculates and visualizes differences between the original images and their reconstructions to assess model's performance
