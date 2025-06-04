# Palm Vein Recognition
Biometric Authentication using Alexnet
###

This project implements a biometric authentication system using convolutional neural networks (CNNs) for two datasets: DB_Vein and FYODB. The system uses a custom AlexNet-inspired architecture to classify vein images for identification purposes.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/IAmFarrokhnejad/Palm-Vein-Recognition.git
   ```
2. Install the required dependencies

## Datasets

- **DB_Vein**: Contains 13,721 vein images from 98 subjects: https://www.kaggle.com/datasets/michaelgoh/contactless-knuckle-palm-print-and-vein-dataset?resource=download
- **FYODB**: Contains 640 (+ 6400 augmented) vein images from 160 subjects: https://fyo.emu.edu.tr/en/download

The datasets should be placed in the following directories:
- `Data/Vein_Dataset/DB_Vein` for DB_Vein
- `Data/FYODB` for FYODB

## Model Architecture

The model is a custom CNN inspired by AlexNet, with the following structure:
- Convolutional layers with ReLU activation and max pooling
- Batch normalization
- Fully connected layers with dropout

The number of output classes differs based on the dataset:
- 98 for DB_Vein
- 160 for FYODB

## Training

To train the model, run:
```
python 1PalmVeinTrain.py
```
This script will:
- Load the datasets
- Train the model on the entire dataset for each (DB_Vein and FYODB)
- Save the trained models as `DB_Vein_model.pth` and `FYODB_model.pth`
- Generate training plots for loss and accuracy

Hyperparameters are fixed for each dataset:
- DB_Vein: learning rate=0.001, batch size=16, dropout=0.2, optimizer=SGD
- FYODB: learning rate=0.005, batch size=16, dropout=0.2, optimizer=SGD

## Prediction

To predict the label of a new image, use:
```
python 2Application.py --dataset [DB_Vein or FYODB] --image path/to/image.png
```
This will load the corresponding model and output the predicted label.


## Credits

Developed by Morteza Farrokhnejad. For questions, contact Morteza.Farrokhnejad@emu.edu.tr .
