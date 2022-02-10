# Emotion Recognition

## 1. Introduction

The algorithm is used to detect 7 kinds of human emotions. The algorithm uses convolutional neural network with dropout as backbone. Throughout the training, fer2013 is used to train the network and data augmentation is performed. We also used techniques like early stopping to increase the performance. In the real-life implementation, we also used cascadeclassifier from opencv to detect human faces.

## 2. Prerequisites

run pip install -r requirements.txt

install fer2013 in the folder 

## 3. Requirements

Python 3.6

Keras 2.4.3

Windows10 20H2

## 3. Training

First run save_image_from_fer.py to generate a folder called dataset. Run train.py and trained models will be stored in trained_models.

## 4. Implementation

Run video_detection.py to implement the detection using camera from your computer.

Note: please fill in the parameter emotion_model_path before implementing.
