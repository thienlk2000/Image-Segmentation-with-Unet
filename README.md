# Image-Segmentation-with-Unet
This repo implement segmentation task with Unet on carvana dataset on Kaggle
## Unet architecture
![file](https://github.com/thienlk2000/Image-Segmentation-with-Unet/blob/main/images/unet.png)

Unet consists 2 parts: down-sample to reduce resolutions and up-sample stage to restore the resolution of the original images

## Dataset.
I use carvana dataset on [Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge) (train.zip) and split test_set(raito=0.2) for validation
This dataset consists only one class is car, so the dimension of output is 1

## Training

Use activation sigmoid to produce probability for 1 class and loss function is binary-cross-entropy
Optimizer is Adam with lr=1e-4

## Result

Train model with 3 epoch and gain 99% accuracy and 98% for dice score 

![file](https://github.com/thienlk2000/Image-Segmentation-with-Unet/blob/main/images/result-unet.JPG)

![file](https://github.com/thienlk2000/Image-Segmentation-with-Unet/blob/main/images/masked_1.png)

![file](https://github.com/thienlk2000/Image-Segmentation-with-Unet/blob/main/images/pred_1.png)
