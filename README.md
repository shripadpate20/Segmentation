# Skin Lesion Segmentation using MobileNet and Custom Decoder

This project implements a deep learning model for segmenting skin lesions in the ISIC 2016 dataset. It uses a MobileNet architecture pre-trained on ImageNet as the encoder and a custom decoder for segmentation.

## Project Overview

- **Dataset**: ISIC 2016
- **Task**: Image Segmentation
- **Model Architecture**: MobileNet (encoder) + Custom Decoder
- **Framework**: PyTorch

## Key Features

1. Utilizes transfer learning with a pre-trained MobileNet encoder
2. Implements a custom decoder for segmentation tasks
3. Explores two training approaches:
   - Freezing encoder weights
   - Fine-tuning the entire model

## Results

### Model with Frozen Encoder Weights

- Mean Dice Score: 0.87
- Mean IoU Score: 0.79
- Average Test Loss: 0.18

### Model with Fine-tuned Weights

- Mean Dice Score: 0.88
- Mean IoU Score: 0.80
- Average Test Loss: 0.21

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy
- Pillow

## Usage

1. Clone the repository
2. Install the required dependencies
3. Prepare the ISIC 2016 dataset
4. Run the `deep_learning_assignment_third.py` script

## Model Architecture

- **Encoder**: Modified MobileNet (pre-trained on ImageNet)
- **Decoder**: Custom architecture with 2D convolution and transposed convolution layers

## Training

The model is trained for 25 epochs using Binary Cross-Entropy loss and the Adam optimizer.

## Evaluation

The model's performance is evaluated using:
- Dice Score
- Intersection over Union (IoU)
- Visual comparison of input images, ground truth masks, and predicted masks

## Future Work

- Experiment with different encoder architectures
- Implement data augmentation techniques
- Explore advanced loss functions for segmentation tasks

## Contributors

- Shripad pate

## Acknowledgments

This project was completed as part of the Deep Learning course at the Indian Institute of Technology, Jodhpur.
