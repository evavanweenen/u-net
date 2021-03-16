import torch 
from torch import nn, optim
from torch.nn import functional as F

from skimage.filters import threshold_otsu

class ConvBlock(nn.Module):
	""" 
	Convolution block for the U-Net architecture 
		consisting of two blocks with a 2D convolution layer, 
		followed by a batch normalization layer, 
		followed by the ReLU function.
	"""
	def __init__(self, in_channels, out_channels, kernel_size):
		super().__init__()

		self.conv_block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True))

	def forward(self, x):
		return self.conv_block(x)

class Down(nn.Module):
	"""
	Down-sampling (encoder) block for the U-Net architecture, consisting of
	a 2D convolution block and a 2D max-pooling layer.
	
	It returns both the convolution block with the max-pooling layer, and
	the convolution block without the max-pooling layer, such that it can
	be concatenated later without max-pooling to the corresponding decoder.
	"""
	def __init__(self, in_channels, out_channels, kernel_size, pool_size):
		super().__init__()
		self.pool = nn.MaxPool2d(pool_size, stride=2)
		self.conv_block = ConvBlock(in_channels, out_channels, kernel_size)

	def forward(self, x):
		return self.pool(self.conv_block(x)), self.conv_block(x)

class Up(nn.Module):
	"""
	Up-sampling (decoder) block for the U-Net architecture, consisting of 
	a 2D convolution block, and a 2D transposed convolution layer.
	
	Besides the previous decoder output, its arguments consist of a
	corresponding encoder block such that it can be concatenated with the 
	2D transposed convolution layer.
	"""
	def __init__(self, in_channels, out_channels, kernel_size, pool_size):
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels, out_channels, pool_size, stride=2)
		self.conv_block = ConvBlock(in_channels, out_channels, kernel_size)

	def forward(self, dec, enc):
		return self.conv_block(torch.cat([self.up(dec), enc], dim=1))

class UNET(nn.Module):
	"""
	U-Net architecture consisting of various Down-sampling and Up-sampling blocks.

	This architecture is used for image segmentation of 2D CT scans. It uses images 
	of (channels x width x height) and outputs masks of the same size (channels x width x height).

	The depth of the network is determined by the length of *channels*. 
	This variable contains the list of filters for all up- and down-sampling blocks, 
	i.e. channels = [c_1,c_2,c_3,...c_n], 
	where the (n-1) down-sampling blocks have filters [c_1,c_2,...,c_{n-1}] respectively,
	and the (n-1) up-sampling blocks have filters [c_{n-1},...,c2,c1] respectively.
	
	The number of filters in the bottleneck is determined by the last number in the list of
	channels, c_n.
	"""
	def __init__(self, channels, kernel_size=(3,3), pool_size=(2,2)):
		super(UNET, self).__init__()
		self.channels = channels

		self.down_convs = nn.ModuleList([]) # down-sampling blocks in the order of propagation
		self.up_convs = nn.ModuleList([]) # up-sampling blocks in *reversed* propagation order

		# down-sampling blocks
		for i in range(len(channels[:-1])):
			in_channels = channels[i-1] if i != 0 else 1
			out_channels = channels[i]
			self.down_convs.append(Down(in_channels, out_channels, kernel_size, pool_size))

		self.bottleneck = ConvBlock(channels[-2], channels[-1], kernel_size)

		# up-sampling block
		for i in range(len(channels[:-1])):
			in_channels = channels[i+1]
			out_channels = channels[i]
			self.up_convs.append(Up(in_channels, out_channels, kernel_size, pool_size))

		self.out_conv = nn.Conv2d(channels[0], 1, kernel_size=1)

	def forward(self, x):
		enc = []

		for i, module in enumerate(self.down_convs):
			x, before_pool = module(x)
			enc.append(before_pool)

		x = self.bottleneck(x)

		for i, module in reversed(list(enumerate(self.up_convs))):
			x = module(x, enc[i])

		x = self.out_conv(x)
		return x

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
		BCE = F.binary_cross_entropy_with_logits(output, target, reduction = self.reduction)
		if self.reduction == 'none':
			BCE = BCE.mean(dim=(-1,-2))
		return self.alpha * (1 - torch.exp(-BCE)) ** self.gamma * BCE

class DiceScore(nn.Module):
	"""
	Dice score = 2*intersection(output,target) / (output + target)

	Before calculating the dice score, a sigmoid activation function is applied on the
	predicted outputs. If the Dice score is not soft, the values are binarized using Otsu's method.

	If the reduction is none, the score for each sample is returned.
	If the reduction is mean, the score is averaged by the batch size.
	"""
	def __init__(self, smooth=1, reduction='mean', soft=False):
		super(DiceScore, self).__init__()
		self.smooth = smooth
		self.reduction = reduction
		self.soft = soft

	def forward(self, output, target):
		""" Calculate DiceScore for each item in batch """
		output = torch.sigmoid(output)

		# binarize
		if not self.soft:
			thresh = [threshold_otsu(o) for o in output.cpu().detach().numpy()]
			output = torch.stack([(output[i] > t).float() for i, t in enumerate(thresh)])

		intersection = (output * target).sum(dim=(-1,-2))

		dice = (2.*intersection + self.smooth) / (output.sum(dim=(-1,-2)) + target.sum(dim=(-1,-2)) + self.smooth)
		
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
	def __init__(self, smooth=1, reduction='mean'):
		super(IoUScore, self).__init__()
		self.smooth = smooth
		self.reduction = reduction

	def forward(self, output, target):
		""" Calculate IoU for each item in batch """
		output = torch.sigmoid(output)

		# binarize
		thresh = [threshold_otsu(o) for o in output.cpu().detach().numpy()]
		output = torch.stack([(output[i] > t).float() for i, t in enumerate(thresh)])

		intersection = (output * target).sum(dim=(-1,-2))
		union = (output + target).sum(dim=(-1,-2)) - intersection

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
	def __init__(self, weight, alpha=1, gamma=2, reduction='mean', smooth=1, soft=True):
		super(FocalDiceLoss, self).__init__()
		self.weight = weight
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = reduction
		self.smooth = smooth
		self.soft = soft

	def forward(self, output, target):
		FL = FocalLoss(self.alpha, self.gamma, self.reduction)(output, target)
		DL = 1 - DiceScore(self.smooth, self.reduction, self.soft)(output, target)
		return FL + self.weight * DL