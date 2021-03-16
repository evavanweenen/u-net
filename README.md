# u-net
U-Net [1] for image segmentation of colon cancer in CT scans
as part of the https://decathlon-10.grand-challenge.org/ task 10

## Introduction
Colon cancer is a disease that is diagnosed in over 1 million people in the world population every year and is one of the deadliest cancers [2]. To aid physicians in monitoring its presence, we use Convolutional Neural Networks to automatically detect the presence, size and location of tumors in the colon from Computer Tomography (CT) imaging.

## Architecture overview (2D)
![alt text](https://github.com/evavanweenen/u-net/blob/main/architecture.png)

## Instructions
Adjust hyper-parameters in `config.py`
* `FILTERS`: list of filters/channels of the U-Net. The last item in the list is the number of filters at the bottleneck (transition between encoder and decoder). 
* Loss function is a combination of Focal loss (with parameters `GAMMA` and `ALPHA`) and the (soft) Dice loss, where `WEIGHT` is the contribution of the Dice loss to the total loss function.
* Hyperparameters are optimized through a grid-search over `PARAM_GRID` with `K`-fold cross-validation, split based on patient ID.

[1] O. Ronneberger, P. Fischer, and  T. Brox. U-net: Convolutional networks for biomedical imagesegmentation. In Medical Image Computing andComputer-Assisted Intervention – MICCAI 2015,pages 234–241, 2015

[2] P. Rawla, T. Sunkara, and A. Barsouk. Epidemiology of colorectal cancer: incidence, mortality, survival,and risk factors.Prz Gastroenterol, 14(2):89–103,2019

[3] T. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar. Focal Loss for Dense Object Detection. In2017 IEEE International Conference on Computer Vision(ICCV), pages 2999–3007, 2017.
