# u-net
U-Net for image segmentation of colon cancer in CT scans
as part of the https://decathlon-10.grand-challenge.org/ task 10

## Instructions
Adjust hyper-parameters in `config.py`
* `FILTERS`: list of filters/channels of the U-Net. The last item in the list is the number of filters at the bottleneck (transition between encoder and decoder). 
* Loss function is a combination of Focal loss (with parameters `GAMMA` and `ALPHA`) and the (soft) Dice loss, where `WEIGHT` is the contribution of the Dice loss to the total loss function.
* Hyperparameters are optimized through a grid-search over `PARAM_GRID` with `K`-fold cross-validation, split based on patient ID.
