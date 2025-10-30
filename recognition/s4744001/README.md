# 2D OASIS Brain MRI segementation using Improved UNet
Andrew Crockford s4744001

## Problem
This project implements an improved 2D UNet model for segmentation of brain MRI slices from the OASIS dataset. The task involves automatically identifying different brain tissue types (gray matter, white matter, and cerebrospinal fluid) from 2D MRI scans.  
The goal is to accurately segment the brain regions with a minimum Dice Similarity Coefficient (DSC) of 0.9 across all labels, demonstrating high precision in medical image segmentation.

## Algorithm
The model used was an improved UNet. A UNet is a convolutional neural network architecture designed for image segmentation tasks, where the goal is to classify each pixel in an image. It follows an encoder–decoder structure where the encoder downsamples the input image through convolutional and pooling layers to extract features, while the decoder upsamples these features to reconstruct a segmentation map. This makes U-Nets effective for medical image segmentation prediction tasks.
- **Encoder:** Captures context using convolutional and pooling layers.
- **Decoder:** Reconstructs the segmentation map using convolutions to recover spatial detail.
- **Output layer:** Uses a 1×1 convolution to predict pixel-wise class probabilities for four tissue classes.

The model is trained using Cross-Entropy Loss, with performance monitored using the Dice coefficient to evaluate overlap between predicted and ground-truth masks.
