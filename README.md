# Cutout

Cutout is a real-time human segmentation Model written in Python using Pytorch, light, stable and accurate.

## Introduction
This software is capable of recognizing
people in real-time with different gestures in a video.

This model uses techniques from MobileNetV2 and UNet, and Dice loss, CrossEntropy loss and SSIM loss are used during training
in order to get plausible results.

The pre-trained model size is only 9.72MB, which means that the model can be easily applied in mobile software.

## Prerequisites
* Python 3.6
* Pytorch 1.4

## Installation
* Clone this repo:
```
git clone https://github.com/BaldwinHe/Cutout
cd Cutout/
```
* Install python requirements:
```
pip install -r requirements.txt
```

## Getting Started
### 0.Download dataset
I use 45986 images from ATR and CIHP dataset to train this model, and 5000 images from CIHP are used for validation.

you can download the dataset from [Look into Person (LIP) ](http://www.sysu-hcp.net/lip/overview.php)
### 1.Open training observer
```
visdom
```
then, open [http://localhost:8097](http://localhost:8097)

### 2.Training
```
python train.py --train_image_dir [path to train image dir] --train_label_dir [path to train label dir] --valid_image_dir [path to valid image dir] --valid_label_dir [path to valid label dir]  
```
you can also change some training hyperparameters:
```
--lr : learning rate
--epoch : epoch number
--log_freq : log frequency
--batch_size : training batch size
--checkpoint : path to checkpoint
```

### 3.Testing
```
python test.py --checkpoint [path to checkpoint]
```

## Results
> video from [Hollywood2](https://www.di.ens.fr/~laptev/actions/hollywood2/) dataset

![Demo](https://github.com/BaldwinHe/DemoLibrary/blob/master/Computer%20Vision/Cutout/cutout.gif)

## Ideas for Future Work
* Weight Pruning
* Introduce NLP techniques to get more plausible and stable video result
* Be able to detect people's back
