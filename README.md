# u-net
U-Net [1] for image segmentation of colon cancer in CT scans
as part of the [Medical Decathlon](https://decathlon-10.grand-challenge.org/) Task 10. 

## Introduction
Colon cancer is a disease that is diagnosed in over 1 million people in the world population every year and is one of the deadliest cancers [2]. To aid physicians in monitoring its presence, we use Convolutional Neural Networks to automatically detect the presence, size and location of tumors in the colon from Computer Tomography (CT) imaging.

## Data
The data consists of 3D CT scans of people with colon cancer, as described in [3], as well as their associated segmentation masks. We consider the 3D images to be batches of 2D images, and for this purpose use a 2D U-Net [1]. To save time during training, we filter out the 2D images with a zero segmentation mask, and split images into folds using their patient ID to avoid data leakage.

## Architecture overview (2D)
We use a 2D U-Net, for which the architecture can be viewed underneath. This is a model that takes as input 2D images, and as output 2D segmentation masks. 
![alt text](https://github.com/evavanweenen/u-net/blob/main/architecture.png)
Examples of predictions can be viewed underneath. Note that the output is the prediction *before* applying a sigmoid function and classification threshold.

## Instructions
Adjust hyper-parameters in `config.py`
* `FILTERS`: list of filters/channels of the U-Net. The last item in the list is the number of filters at the bottleneck (transition between encoder and decoder). 
* Loss function is a combination of Focal loss [4], with parameters `GAMMA` and `ALPHA`, and the (soft) Dice loss, where `WEIGHT` is the contribution of the Dice loss to the total loss function.
* Hyperparameters are optimized through a grid-search over `PARAM_GRID` with `K`-fold cross-validation, split based on patient ID.

## References
[1] O. Ronneberger, P. Fischer, and  T. Brox. U-net: Convolutional networks for biomedical imagesegmentation. In Medical Image Computing andComputer-Assisted Intervention – MICCAI 2015,pages 234–241, 2015

[2] P. Rawla, T. Sunkara, and A. Barsouk. Epidemiology of colorectal cancer: incidence, mortality, survival,and risk factors. Prz Gastroenterol, 14(2):89–103,2019

[3] Amber L. Simpson, Michela Antonelli, Spyridon Bakas et al. A large annotated medical image dataset for the development and evaluation of segmentation algorithms. In CoRR, 2019. ArXiv: [1902.09063](https://arxiv.org/abs/1902.09063)

[4] T. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar. Focal Loss for Dense Object Detection. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 2999–3007, 2017.
