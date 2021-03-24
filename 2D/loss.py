import torch 
from torch import nn, optim
from torch.nn import functional as F

from torchvision.ops import sigmoid_focal_loss

import numpy as np

class FocalLoss(nn.modules.loss._WeightedLoss):
	""" 
	Focal Loss as described by https://arxiv.org/1708.02002 
	with parameters alpha describing the weighted update according to class imbalance
	and gamma the focus parameter, reducing the loss contribution from easy examples.

	FL(p_t) = - alpha_t (1-p_t)^gamma log(p_t) with p_t = exp(-BCE)

	If the reduction is none, the loss for each sample is returned.
	If the reduction is mean, the loss is averaged by the batch size.
	"""
	def __init__(self, alpha=1, gamma=2, reduction='mean'):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = reduction

	def forward(self, output, target):
		fl = sigmoid_focal_loss(output, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
		if self.reduction == 'none':
			fl = fl.mean(dim=(-1,-2)) # average over the image
		return fl

class DiceScore(nn.Module):
	"""
	Dice score = 2*intersection(output,target) / (output + target)

	Before calculating the dice score, a sigmoid activation function is applied on the
	predicted outputs. If the Dice score is not soft, the values are binarized using Otsu's method.

	If the reduction is none, the score for each sample is returned.
	If the reduction is mean, the score is averaged by the batch size.
	"""
	def __init__(self, smooth=1, reduction='mean', soft=False, dim=2, threshold=.5):
		super(DiceScore, self).__init__()
		self.smooth = smooth
		self.reduction = reduction
		self.soft = soft
		self.dim = dim
		self.aggdim = tuple(-np.arange(1,self.dim+1))
		self.threshold = threshold

	def forward(self, output, target):
		""" Calculate DiceScore for each item in batch """
		output = torch.sigmoid(output)

		# binarize
		if not self.soft:
			output = (output > self.threshold).float()

		intersection = (output * target).sum(dim=self.aggdim)

		dice = (2.*intersection + self.smooth) / (output.sum(dim=self.aggdim) + target.sum(dim=self.aggdim) + self.smooth)
		
		if self.reduction == 'none':
			return dice
		elif self.reduction == 'mean':
			return dice.mean()

class IoUScore(nn.Module):
	"""
	IoU score = intersection(output,target) / union(output,target)

	Before calculating the dice score, a sigmoid activation function is applied on the
	predicted outputs, after which the values are binarized using Otsu's method.

	If the reduction is none, the score for each sample is returned.
	If the reduction is mean, the score is averaged by the batch size.
	"""
	def __init__(self, smooth=1, reduction='mean', dim=2, threshold=.5):
		super(IoUScore, self).__init__()
		self.smooth = smooth
		self.reduction = reduction
		self.dim = dim
		self.aggdim = tuple(-np.arange(1,self.dim+1))
		self.threshold = threshold

	def forward(self, output, target):
		""" Calculate IoU for each item in batch """
		output = torch.sigmoid(output)

		# binarize
		output = (output > self.threshold).float()

		intersection = (output * target).sum(dim=self.aggdim)
		union = (output + target).sum(dim=self.aggdim) - intersection

		iou = (intersection + self.smooth) / (union + self.smooth)

		if self.reduction == 'none':
			return iou
		elif self.reduction == 'mean':
			return iou.mean()

class FocalDiceLoss(nn.Module):
	""" 
	Loss function consisting of a weighted average of the Focal Loss (see above), and
	the Dice loss (see above). 

	L = FL + weight*(1-dice)

	If the reduction is none, the loss for each sample is returned.
	If the reduction is mean, the loss is averaged by the batch size.
	"""
	def __init__(self, weight, alpha=1, gamma=2, reduction='mean', smooth=1, soft=True, dim=2, threshold=.5):
		super(FocalDiceLoss, self).__init__()
		self.weight = weight
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = reduction
		self.smooth = smooth
		self.soft = soft
		self.dim = dim
		self.threshold = threshold

	def forward(self, output, target):
		FL = FocalLoss(self.alpha, self.gamma, self.reduction)(output, target)
		DL = 1 - DiceScore(self.smooth, self.reduction, self.soft, self.dim, self.threshold)(output, target)
		return FL + self.weight * DL