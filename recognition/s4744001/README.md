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

## Data Preprocessing
The data was processed to prepare the 2D OASIS brain MRI slices for segmentation. The steps taken were pairing image/mask PNGs, loading them in grayscale, applying z-score normalisation to images, remapping mask labels to class IDs, resizing images to 256×256, and returning the PyTorch tensors for training, validation, and testing. 

## Training and Evaluation
To train the model run:
``` 
python train.py
```
- The model learns from the training dataset using CrossEntropyLoss.
- Training runs for 20 epochs.
- The Dice coefficient is calculated after each epoch.
- Validation is performed at the end of every epoch.
- Training and validation metrics are logged for later analysis in predict.py
- The final trained model is saved as trained_model.pt, and logs are saved to training_logs.pth.

Example output while training:
``` 
===== Epoch 19/20 =====
[Train] Batch 10/2416 Loss=0.0188
[Val] Batch 2290/2416 Loss=0.0187 Dice=0.9642
[Epoch 19] TrainLoss=0.0232 | ValLoss=0.0220 | TrainDice=0.9615 | ValDice=0.9607
```

After training, run:
``` 
python predict.py
```
- Loads the trained model from trained_model.pt
- Evaluates performance on validation and test sets
- Generates training curves and example segmentation visualizations

These are the following outputs:
```
[Val]  Loss=0.0235  MeanDice=0.9594  PerClass=(0.99897397 0.92523533 0.9455089  0.967722)
[Test] Loss=0.0238  MeanDice=0.9612  PerClass= (0.9989563  0.93119913 0.94583994 0.9686278)
```

**Figure 1: Example segementation results from UNet model**

<img width="600" height="500" alt="image" src="https://github.com/Acrockford1202/PatternAnalysis-2025-s4744001---Task-1/blob/topic-recognition/output_sample1.png" />
<img width="600" height="500" alt="image" src="https://github.com/Acrockford1202/PatternAnalysis-2025-s4744001---Task-1/blob/topic-recognition/output_sample2.png" />

**Figure 2: Training loss vs number of epochs**


<img width="600" height="500" alt="image" src="https://github.com/Acrockford1202/PatternAnalysis-2025-s4744001---Task-1/blob/topic-recognition/training_curves_loss.png" />

**Figure 3: Dice score vs number of epochs**


<img width="600" height="500" alt="image" src="https://github.com/Acrockford1202/PatternAnalysis-2025-s4744001---Task-1/blob/topic-recognition/training_curves_dice.png" />

## Dependencies 
Assume the user has the latest version of python installed. The project requires the following to be installed:
```
pip install torch
pip install torchvision
pip install torchaudio
pip install numpy
pip install opencv-python
pip install matplotlib
```
