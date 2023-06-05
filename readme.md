# Problem statement
This repository contains the code and resources for solving the Airbus Ship Detection Challenge from Kaggle. The challenge involves detecting ships in satellite images.

# Data Overview
The dataset used in this project can be obtained from the [Airbus Ship Detection Challenge](https://www.kaggle.com/competitions/airbus-ship-detection/data) on Kaggle. It consists of a large number of satellite images in which ships are labeled. The images are provided along with corresponding binary masks indicating the ship locations.

## Data Preprocessing

### Masks encoding

To convert the RLE-encoded masks to pixel masks, I implemented the following steps:

1. Parse the RLE-encoded strings for each image's mask.
2. Decode the RLE strings to obtain the pixel positions of the ship regions.
3. Generate binary masks by assigning a value of 1 to pixels within ship regions and 0 to pixels outside ship regions.
   
By performing this conversion, I transformed the masks into a format compatible with image segmentation models.

### Data sampling

The dataset provided for the task suffers from class imbalance, as the number of ship instances is significantly smaller compared to the background (non-ship) instances. To address this issue, I employed data sampling before training.

![stats1](/assets/sampled_datatset.png)

# Model training

U-Net was used to solve this problem as it is one of the best neural network architectures for image segmentation.

## Loss function 

__Combo loss__ is a weighted sum of Dice loss and a modified cross entropy. It attempts to leverage the flexibility of Dice loss of class imbalance and at same time
use cross-entropy for curve smoothing.

$DL(y, \hat p) = 1 - \frac{2y\hat p+1}{y+\hat p+1}$

$L_{m-bce}=-\frac{1}{N}\sum_{i=1}(\beta (y-log(\hat y)) + (1-\beta)(1-y)*log(1-\hat y))$

$CL(y, \hat) = \alpha L_{m-bce} - (1-\alpha)DL(y, \hat p)$

## Training results

The results of the training are shown in the graph below.

![stats1](/assets/stats.png)

Below are examples of model prediction for test images.

![stats1](/assets/r1.png)

![stats1](/assets/r2.png)

We can see that the model generally performs quite well on test data, with the exception of images with big waves.


# User guide

## Executing with requirements.txt

1. Create vertual environment

        python3 -m venv venv

2. Activate the environment

        source venv/bin/activate    

3. install dependencies from requirements.txt

        pip3 install -r requirements.txt

4. run inference.py by specifing flag "--input" and path to the data folder with images.

        python3 inference.py --input "folder with images"


## Prediction examples


Input image             |  Predicted mask
:-------------------------:|:-------------------------:
![](/assets/img1.jpg)  |  ![](/assets/img1_prediction.png)


