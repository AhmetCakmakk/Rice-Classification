# Rice Image Classification with Convolutional Neural Networks (CNN)

This project implements a Convolutional Neural Network (CNN) to classify different varieties of rice grains using a comprehensive image dataset. The primary goal is to leverage deep learning techniques to accurately distinguish between five different types of rice based on their visual characteristics.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
  - [Data Splitting](#data-splitting)
  - [Data Visualization](#data-visualization)
- [Model Architecture](#model-architecture)
  - [Layer Details](#layer-details)
  - [Hyperparameters](#hyperparameters)
- [Training](#training)
  - [Callbacks](#callbacks)
  - [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
  - [Training and Validation Performance](#training-and-validation-performance)
  - [Confusion Matrix](#confusion-matrix)
  - [Classification Report](#classification-report)
- [Model Deployment](#model-deployment)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

Image classification is a crucial task in computer vision, and it has wide applications in areas like agriculture, healthcare, and industry. In this project, I build a deep learning model that can classify images of rice grains into five distinct categories. The classification is based solely on the visual features of the rice grains, using a dataset provided by Kaggle.

The project involves several stages:
1. Data collection and preprocessing.
2. CNN model design and training.
3. Model evaluation and performance analysis.
4. Final model saving and potential deployment strategies.

## Dataset

The dataset used for this project is the 'Rice Image Dataset', sourced from Kaggle. It consists of 75,000 images distributed across five classes:
- **Arborio**: 15,000 images
- **Basmati**: 15,000 images
- **Ipsala**: 15,000 images
- **Jasmine**: 15,000 images
- **Karacadag**: 15,000 images

Each image is a high-resolution close-up of a rice grain, and the dataset is well-balanced with an equal number of samples in each class.

### Dataset Distribution

The dataset is divided into three subsets:
- **Training Set (80%)**: 60,000 images
- **Validation Set (10%)**: 7,500 images
- **Test Set (10%)**: 7,500 images

This distribution ensures that the model can generalize well to unseen data.


## Requirements

Ensure you have the following libraries installed:

!pip install split-folders  
!pip install plotly  
!pip install tensorflow   
!pip install scikit-learn  
!pip install matplotlib  
!pip install seaborn  

## Data Preparation

### Data Splitting

The original dataset is split into training, validation, and test sets using the splitfolders library. The data is split in an 80:10:10 ratio to ensure that the model is trained on a large portion of the data while still having separate sets for validation and testing.

splitfolders.ratio(data_dir, output=output_dir, seed=1337, ratio=(0.8, 0.1, 0.1))


### Data Visualization

To understand the data distribution and characteristics, we visualize the number of images in each class and display random samples from each class.

## Model Architecture

The Convolutional Neural Network (CNN) model used in this project consists of several layers designed to extract and learn features from the images.

### Layer Details

1. **Rescaling Layer**: Normalizes pixel values between 0 and 1.
2. **Convolutional Layers**: Extracts features from the input images using multiple filters.
   - Conv2D(16 filters, kernel size 3x3, activation='relu')
   - Conv2D(32 filters, kernel size 3x3, activation='relu')
   - Conv2D(64 filters, kernel size 3x3, activation='relu')
3. **MaxPooling Layers**: Reduces the spatial dimensions of the feature maps.
4. **Flatten Layer**: Converts the 2D matrix data to a 1D vector.
5. **Dense Layers**: Fully connected layers for classification.
   - Dense(128 units, activation='relu')
   - Dropout(0.2) for regularization.
   - Dense(num_classes, activation='softmax') for the output layer.

### Hyperparameters

The key hyperparameters used in this model include:
- **Optimizer**: Adam with a learning rate of 0.0001
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 64
- **Epochs**: 10

## Training

The model is trained on the training dataset with validation performed on the validation set.

### Callbacks

Two important callbacks are used during training:
- **Early Stopping**: Stops training when validation accuracy does not improve for 5 consecutive epochs.
- **Model Checkpointing**: Saves the best-performing model based on validation accuracy.

### Evaluation Metrics
- **Accuracy**: Evaluated on both training and validation datasets.
- **Confusion Matrix**: Provides a detailed breakdown of the classification results.
- **Classification Report**: Includes precision, recall, and F1-score for each class.

## Results

### Training and Validation Performance

The model achieves high accuracy on both the training and validation datasets. The results indicate that the model is well-trained without overfitting.

### Confusion Matrix

The confusion matrix visualizes the performance of the model across all classes, showing the true positive, false positive, and false negative counts.

### Classification Report

The classification report includes detailed metrics such as precision, recall, and F1-score for each rice variety.

## Model Deployment

The trained model is saved as `CNN_model.h5` for future use. This model can be deployed in various applications such as mobile apps or web services to classify rice images in real-time.

## Usage

To use the model, load it using TensorFlow and pass an image or a batch of images for prediction. The model will output the predicted class labels.

```python
from tensorflow.keras.models import load_model

model = load_model('CNN_model.h5')
predictions = model.predict(new_images)  
```

## Conclusion
This project successfully demonstrates the power of Convolutional Neural Networks in image classification tasks, specifically in distinguishing different types of rice grains. With further tuning and expansion, this model could serve as the foundation for more advanced agricultural image classification systems.


