# modules.py
# 
# The modules in this file include layers like convolutions, residual blocks and coupling layers.

import copy
import math
import numpy as np
import scipy
import torch
import commons

from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm
from commons import init_weights, get_padding
from transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1

class LayerNorm(nn.Module):

	"""
	Custom Layer Normalization implementation.
	
	Args:
		channels (int): Number of channels in the input tensor.
		eps (float, optional): A small value to avoid division by zero. Default is 1e-5.
	"""

	def __init__(self, channels, eps=1e-5):

		super().__init__()

		self.channels = channels
		self.eps = eps
		self.gamma = nn.Parameter(torch.ones(channels))
		self.beta = nn.Parameter(torch.zeros(channels))

	def forward(self, x):

		"""
		Forward pass of LayerNorm.

		Args:
			x (torch.Tensor): The input tensor.

		Returns:
			torch.Tensor: The normalized output tensor.
		"""

		x = x.transpose(1, -1)
		x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)

		return x.transpose(1, -1)

class ConvReluNorm(nn.Module):

	"""
	Convolutional layer followed by ReLU activation and Layer Normalization.
	
	Args:
		in_channels (int): The number of input channels.
		hidden_channels (int): The number of hidden channels.
		out_channels (int): The number of output channels.
		kernel_size (int): The size of the convolution kernel.
		n_layers (int): The number of convolutional layers.
		p_dropout (float): The probability of dropout.
	"""

	def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):

		super().__init__()
	
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.n_layers = n_layers
		self.p_dropout = p_dropout
	
		assert n_layers > 1, "Number of layers should be larger than 0."

		self.conv_layers = nn.ModuleList()
		self.norm_layers = nn.ModuleList()
		self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2))
		self.norm_layers.append(LayerNorm(hidden_channels))
		self.relu_drop = nn.Sequential(nn.ReLU(),nn.Dropout(p_dropout))

		for _ in range(n_layers - 1):

			self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
			self.norm_layers.append(LayerNorm(hidden_channels))

		self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
		self.proj.weight.data.zero_()
		self.proj.bias.data.zero_()

	def forward(self, x, x_mask):

		"""
		Forward pass through the ConvReluNorm block.

		Args:
			x (torch.Tensor): The input tensor.
			x_mask (torch.Tensor): The mask for padding or other conditions.

		Returns:
			torch.Tensor: The processed output tensor.
		"""

		x_org = x

		for i in range(self.n_layers):

			x = self.conv_layers[i](x * x_mask)
			x = self.norm_layers[i](x)
			x = self.relu_drop(x)

		x = x_org + self.proj(x)

		return x * x_mask

class DDSConv(nn.Module):

	"""
	Dilated and Depth-Separable Convolution.

	Args:
		channels (int): The number of input/output channels.
		kernel_size (int): The size of the convolution kernel.
		n_layers (int): The number of layers in the network.
		p_dropout (float, optional): The probability of dropout.
	"""

	def __init__(self, channels, kernel_size, n_layers, p_dropout=0.):

		super().__init__()

		self.channels = channels
		self.kernel_size = kernel_size
		self.n_layers = n_layers
		self.p_dropout = p_dropout
		self.drop = nn.Dropout(p_dropout)
		self.convs_sep = nn.ModuleList()
		self.convs_1x1 = nn.ModuleList()
		self.norms_1 = nn.ModuleList()
		self.norms_2 = nn.ModuleList()

		for i in range(n_layers):

			dilation = kernel_size ** i

			padding = (kernel_size * dilation - dilation) // 2

			self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding))
			self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
			self.norms_1.append(LayerNorm(channels))
			self.norms_2.append(LayerNorm(channels))

	def forward(self, x, x_mask, g=None):

		"""
		Forward pass through the DDSConv block.

		Args:
			x (torch.Tensor): The input tensor.
			x_mask (torch.Tensor): The mask for padding or other conditions.
			g (torch.Tensor, optional): A conditioning tensor.

		Returns:
			torch.Tensor: The processed output tensor.
		"""

		if g is not None:

			x = x + g

		for i in range(self.n_layers):

			y = self.convs_sep[i](x * x_mask)
			y = self.norms_1[i](y)

			y = F.gelu(y)

			y = self.convs_1x1[i](y)
			y = self.norms_2[i](y)

			y = F.gelu(y)

			y = self.drop(y)

			x = x + y

		return x * x_mask

class WN(torch.nn.Module):

	"""
	WaveNet style residual network.

	Args:
		hidden_channels (int): The number of hidden channels in the network.
		kernel_size (int): The size of the kernel for convolutions.
		dilation_rate (int): The dilation rate for convolutions.
		n_layers (int): The number of layers in the network.
		gin_channels (int, optional): Number of conditioning channels.
		p_dropout (float, optional): The probability of dropout.
	"""

	def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):

		super(WN, self).__init__()

		assert(kernel_size % 2 == 1)

		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.dilation_rate = dilation_rate
		self.n_layers = n_layers
		self.gin_channels = gin_channels
		self.p_dropout = p_dropout
		self.in_layers = torch.nn.ModuleList()
		self.res_skip_layers = torch.nn.ModuleList()
		self.drop = nn.Dropout(p_dropout)

		if gin_channels != 0:

			cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)

			self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

		for i in range(n_layers):

			dilation = dilation_rate ** i

			padding = int((kernel_size * dilation - dilation) / 2)

			in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,dilation=dilation, padding=padding)
			in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')

			self.in_layers.append(in_layer)

			if i < n_layers - 1:

				res_skip_channels = 2 * hidden_channels

			else:

				res_skip_channels = hidden_channels

			res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
			res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')

			self.res_skip_layers.append(res_skip_layer)

	def forward(self, x, x_mask, g=None, **kwargs):

		"""
		Forward pass through the WN block.

		Args:
			x (torch.Tensor): The input tensor.
			x_mask (torch.Tensor): The mask for padding or other conditions.
			g (torch.Tensor, optional): A conditioning tensor.

		Returns:
			torch.Tensor: The processed output tensor.
		"""

		output = torch.zeros_like(x)

		n_channels_tensor = torch.IntTensor([self.hidden_channels])

		if g is not None:

			g = self.cond_layer(g)

		for i in range(self.n_layers):

			x_in = self.in_layers[i](x)

			if g is not None:

				cond_offset = i * 2 * self.hidden_channels

				g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]

			else:

				g_l = torch.zeros_like(x_in)

			acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
			acts = self.drop(acts)

			res_skip_acts = self.res_skip_layers[i](acts)

			if i < self.n_layers - 1:

				res_acts = res_skip_acts[:, :self.hidden_channels, :]

				x = (x + res_acts) * x_mask

				output = output + res_skip_acts[:, self.hidden_channels:, :]

			else:

				output = output + res_skip_acts

		return output * x_mask

	def remove_weight_norm(self):

		"""
		Remove weight normalization from the layers.
		"""

		if self.gin_channels != 0:

			torch.nn.utils.remove_weight_norm(self.cond_layer)

		for l in self.in_layers:

			torch.nn.utils.remove_weight_norm(l)

		for l in self.res_skip_layers:

			torch.nn.utils.remove_weight_norm(l)

class ResBlock1(torch.nn.Module):

	"""
	Residual block with three dilated convolution layers followed by another set
	of convolutions. The output is added to the input for residual learning.

	Args:
		channels (int): Number of channels for input/output.
		kernel_size (int): Kernel size for the convolutions.
		dilation (tuple): List of dilation rates for convolutions.
	"""

	def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):

		super(ResBlock1, self).__init__()

		self.convs1 = nn.ModuleList([
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],padding=get_padding(kernel_size, dilation[0]))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],padding=get_padding(kernel_size, dilation[1]))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],padding=get_padding(kernel_size, dilation[2])))
		])

		self.convs1.apply(init_weights)

		self.convs2 = nn.ModuleList([
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,padding=get_padding(kernel_size, 1))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,padding=get_padding(kernel_size, 1))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,padding=get_padding(kernel_size, 1)))
		])
		
		self.convs2.apply(init_weights)

	def forward(self, x, x_mask=None):

		"""
		Forward pass for ResBlock1. Applies convolutions and residual connections.

		Args:
			x (torch.Tensor): Input tensor.
			x_mask (torch.Tensor, optional): Mask to apply on input tensor.

		Returns:
			torch.Tensor: Output tensor after applying convolutions and residual addition.
		"""

		for c1, c2 in zip(self.convs1, self.convs2):

			xt = F.leaky_relu(x, LRELU_SLOPE)

			if x_mask is not None:

				xt = xt * x_mask

			xt = c1(xt)
			xt = F.leaky_relu(xt, LRELU_SLOPE)

			if x_mask is not None:

				xt = xt * x_mask

			xt = c2(xt)

			x = xt + x

		if x_mask is not None:

			x = x * x_mask

		return x

	def remove_weight_norm(self):

		"""
		Remove weight normalization from all convolution layers.
		"""

		for l in self.convs1:

			remove_weight_norm(l)

		for l in self.convs2:

			remove_weight_norm(l)

class ResBlock2(torch.nn.Module):

	"""
	Residual block with two dilated convolution layers.

	Args:
		channels (int): Number of channels for input/output.
		kernel_size (int): Kernel size for the convolutions.
		dilation (tuple): List of dilation rates for convolutions.
	"""

	def __init__(self, channels, kernel_size=3, dilation=(1, 3)):

		super(ResBlock2, self).__init__()

		self.convs = nn.ModuleList([
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],padding=get_padding(kernel_size, dilation[0]))),
			weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],padding=get_padding(kernel_size, dilation[1]))),
		])

		self.convs.apply(init_weights)

	def forward(self, x, x_mask=None):

		"""
		Forward pass for ResBlock2. Applies convolutions and residual connections.

		Args:
			x (torch.Tensor): Input tensor.
			x_mask (torch.Tensor, optional): Mask to apply on input tensor.

		Returns:
			torch.Tensor: Output tensor after applying convolutions and residual addition.
		"""

		for c in self.convs:

			xt = F.leaky_relu(x, LRELU_SLOPE)

			if x_mask is not None:

				xt = xt * x_mask

			xt = c(xt)

			x = xt + x

		if x_mask is not None:

			x = x * x_mask

		return x

	def remove_weight_norm(self):

		"""
		Remove weight normalization from all convolution layers.
		"""

		for l in self.convs:

			remove_weight_norm(l)

class Log(nn.Module):

	"""
	Logarithmic transformation, with the option to reverse it.

	Args:
		None
	"""

	def forward(self, x, x_mask, reverse=False, **kwargs):

		"""
		Forward pass for Log transformation.

		Args:
			x (torch.Tensor): Input tensor.
			x_mask (torch.Tensor): Mask to apply on input tensor.
			reverse (bool): If True, apply the inverse transformation (exponential).

		Returns:
			torch.Tensor: Transformed output tensor.
			torch.Tensor: Log determinant for Jacobian.
		"""

		if not reverse:

			y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask

			logdet = torch.sum(-y, [1, 2])

			return y, logdet

		else:

			x = torch.exp(x) * x_mask

			return x

class Flip(nn.Module):

	"""
	Flip operation on the input tensor, with the option to reverse the operation.

	Args:
		None
	"""

	def forward(self, x, *args, reverse=False, **kwargs):

		"""
		Forward pass for Flip operation.

		Args:
			x (torch.Tensor): Input tensor.
			reverse (bool): If True, apply the inverse of the flip (flip again).

		Returns:
			torch.Tensor: Transformed output tensor.
			torch.Tensor: Log determinant for Jacobian.
		"""

		x = torch.flip(x, [1])

		if not reverse:

			logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)

			return x, logdet

		else:

			return x

class ElementwiseAffine(nn.Module):

	"""
	Elementwise affine transformation.

	Args:
		channels (int): Number of input channels.
	"""

	def __init__(self, channels):

		super().__init__()

		self.channels = channels
		self.m = nn.Parameter(torch.zeros(channels, 1))
		self.logs = nn.Parameter(torch.zeros(channels, 1))

	def forward(self, x, x_mask, reverse=False, **kwargs):

		"""
		Forward pass for ElementwiseAffine transformation.

		Args:
			x (torch.Tensor): Input tensor.
			x_mask (torch.Tensor): Mask to apply on input tensor.
			reverse (bool): If True, apply the inverse transformation.

		Returns:
			torch.Tensor: Transformed output tensor.
		"""

		if not reverse:

			y = self.m + torch.exp(self.logs) * x
			y = y * x_mask

			logdet = torch.sum(self.logs * x_mask, [1, 2])

			return y, logdet

		else:

			x = (x - self.m) * torch.exp(-self.logs) * x_mask

			return x

class ResidualCouplingLayer(nn.Module):

	"""
	Residual coupling layer for normalizing flows, applying a transformation to half of the channels.

	Args:
		channels (int): Number of input channels.
		hidden_channels (int): Number of hidden channels for intermediate layers.
		kernel_size (int): Kernel size for convolutions.
		dilation_rate (int): Dilation rate for convolutions.
		n_layers (int): Number of layers in the network.
		p_dropout (float, optional): Probability of dropout in the layers.
		gin_channels (int, optional): Conditioning channels for the network.
		mean_only (bool, optional): If True, only the mean part of the transformation is learned.
	"""

	def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):

		assert channels % 2 == 0, "channels should be divisible by 2"

		super().__init__()

		self.channels = channels
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.dilation_rate = dilation_rate
		self.n_layers = n_layers
		self.half_channels = channels // 2
		self.mean_only = mean_only
		self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
		self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
		self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
		self.post.weight.data.zero_()
		self.post.bias.data.zero_()

	def forward(self, x, x_mask, g=None, reverse=False):

		"""
		Forward pass for ResidualCouplingLayer.

		Args:
			x (torch.Tensor): Input tensor.
			x_mask (torch.Tensor): Mask to apply on input tensor.
			g (torch.Tensor, optional): Conditioning tensor.
			reverse (bool): If True, apply the inverse transformation.

		Returns:
			torch.Tensor: Transformed output tensor.
			torch.Tensor: Log determinant for Jacobian.
		"""

		x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

		h = self.pre(x0) * x_mask
		h = self.enc(h, x_mask, g=g)

		stats = self.post(h) * x_mask

		if not self.mean_only:

			m, logs = torch.split(stats, [self.half_channels] * 2, 1)

		else:

			m = stats

			logs = torch.zeros_like(m)

		if not reverse:

			x1 = m + x1 * torch.exp(logs) * x_mask

			x = torch.cat([x0, x1], 1)

			logdet = torch.sum(logs, [1, 2])

			return x, logdet

		else:

			x1 = (x1 - m) * torch.exp(-logs) * x_mask

			x = torch.cat([x0, x1], 1)

			return x

class ConvFlow(nn.Module):

	"""
	Convolutional normalizing flow, based on a coupling layer and convolution layers.

	Args:
		in_channels (int): Number of input channels.
		filter_channels (int): Number of filter channels.
		kernel_size (int): Kernel size for convolutions.
		n_layers (int): Number of layers in the network.
		num_bins (int, optional): Number of bins for rational quadratic splines.
		tail_bound (float, optional): Bound for the tail of the rational quadratic spline.
	"""

	def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):

		super().__init__()

		self.in_channels = in_channels
		self.filter_channels = filter_channels
		self.kernel_size = kernel_size
		self.n_layers = n_layers
		self.num_bins = num_bins
		self.tail_bound = tail_bound
		self.half_channels = in_channels // 2
		self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
		self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.)
		self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
		self.proj.weight.data.zero_()
		self.proj.bias.data.zero_()

	def forward(self, x, x_mask, g=None, reverse=False):

		"""
		Forward pass for ConvFlow.

		Args:
			x (torch.Tensor): Input tensor.
			x_mask (torch.Tensor): Mask to apply on input tensor.
			g (torch.Tensor, optional): Conditioning tensor.
			reverse (bool): If True, apply the inverse transformation.

		Returns:
			torch.Tensor: Transformed output tensor.
			torch.Tensor: Log determinant for Jacobian.
		"""

		x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

		h = self.pre(x0)
		h = self.convs(h, x_mask, g=g)
		h = self.proj(h) * x_mask

		b, c, t = x0.shape

		h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

		unnormalized_widths = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
		unnormalized_heights = h[..., self.num_bins:2 * self.num_bins] / math.sqrt(self.filter_channels)
		unnormalized_derivatives = h[..., 2 * self.num_bins:]

		x1, logabsdet = piecewise_rational_quadratic_transform(
			x1,
			unnormalized_widths,
			unnormalized_heights,
			unnormalized_derivatives,
			inverse=reverse,
			tails='linear',
			tail_bound=self.tail_bound
		)

		x = torch.cat([x0, x1], 1) * x_mask

		logdet = torch.sum(logabsdet * x_mask, [1, 2])

		if not reverse:

			return x, logdet

		else:

			return x
