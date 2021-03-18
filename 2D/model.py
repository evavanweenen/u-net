import torch 
from torch import nn, optim
from torch.nn import functional as F

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