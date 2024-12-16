# commons.py
#
# This file contains utility functions that are frequently used in the VITS model.
# These include functions for weight initialization, padding, timing signals, and operations related to the model's training process.

import math
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

def init_weights(m, mean=0.0, std=0.01):

	"""
	Initialize the weights of a given layer (Conv layers in this case) with a normal distribution.
	"""

	classname = m.__class__.__name__

	if classname.find("Conv") != -1:

		m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):

	"""
	Compute padding size for a convolution layer given kernel size and dilation factor.
	"""

	return int((kernel_size * dilation - dilation) / 2)

def convert_pad_shape(pad_shape):

	"""
	Convert a list of padding shapes into a flattened format for torch.
	"""

	l = pad_shape[::-1]

	pad_shape = [item for sublist in l for item in sublist]

	return pad_shape

def intersperse(lst, item):

	"""
	Inserts the specified item between each element of the input list.
	"""

	result = [item] * (len(lst) * 2 + 1)

	result[1::2] = lst

	return result

def kl_divergence(m_p, logs_p, m_q, logs_q):

	"""
	Computes the Kullback-Leibler (KL) divergence between two distributions P and Q.
	"""

	kl = (logs_q - logs_p) - 0.5

	kl += 0.5 * (torch.exp(2. * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2. * logs_q)

	return kl

def rand_gumbel(shape):

	"""
	Sample from the Gumbel distribution to generate random noise values.
	Protects against overflow during sampling.
	"""

	uniform_samples = torch.rand(shape) * 0.99998 + 0.00001

	return -torch.log(-torch.log(uniform_samples))

def rand_gumbel_like(x):

	"""
	Generates Gumbel noise of the same shape as the given tensor.
	"""

	g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)

	return g

def slice_segments(x, ids_str, segment_size=4):

	"""
	Slice segments from the input tensor `x` based on the starting indices `ids_str` and segment size.
	"""

	ret = torch.zeros_like(x[:, :, :segment_size])

	for i in range(x.size(0)):

		idx_str = ids_str[i]
		idx_end = idx_str + segment_size

		ret[i] = x[i, :, idx_str:idx_end]

	return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):

	"""
	Slice random segments from the input tensor `x`, with each segment having a size of `segment_size`.
	"""

	b, d, t = x.size()

	if x_lengths is None:

		x_lengths = t

	ids_str_max = x_lengths - segment_size + 1
	ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)

	ret = slice_segments(x, ids_str, segment_size)

	return ret, ids_str

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):

	"""
	Generates 1D timing signal with sinusoidal functions for positional encoding.
	"""

	position = torch.arange(length, dtype=torch.float)

	num_timescales = channels // 2

	log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1))

	inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
	
	scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)

	signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
	signal = F.pad(signal, [0, 0, 0, channels % 2])
	signal = signal.view(1, channels, length)

	return signal

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):

	"""
	Adds the 1D timing signal to the input tensor `x`.
	"""

	b, channels, length = x.size()

	signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)

	return x + signal.to(dtype=x.dtype, device=x.device)

def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):

	"""
	Cats the 1D timing signal to the input tensor `x`.
	"""

	b, channels, length = x.size()

	signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)

	return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)

def subsequent_mask(length):

	"""
	Creates a subsequent mask for the transformer model.
	"""

	mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)

	return mask

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):

	"""
	Performs a fused operation: add two inputs, apply tanh and sigmoid activations, then multiply the results.
	"""

	n_channels_int = n_channels[0]

	in_act = input_a + input_b

	t_act = torch.tanh(in_act[:, :n_channels_int, :])
	s_act = torch.sigmoid(in_act[:, n_channels_int:, :])

	acts = t_act * s_act

	return acts

def shift_1d(x):

	"""
	Shifts the tensor `x` by one position in the 1D sequence.
	"""

	x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]

	return x

def sequence_mask(length, max_length=None):

	"""
	Generates a mask for a sequence based on the lengths provided.
	"""

	if max_length is None:

		max_length = length.max()

	x = torch.arange(max_length, dtype=length.dtype, device=length.device)

	return x.unsqueeze(0) < length.unsqueeze(1)

def generate_path(duration, mask):

	"""
	Generates a path for duration alignment.
	"""

	device = duration.device

	b, _, t_y, t_x = mask.shape

	cum_duration = torch.cumsum(duration, -1)
	cum_duration_flat = cum_duration.view(b * t_x)

	path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
	path = path.view(b, t_x, t_y)
	path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
	path = path.unsqueeze(1).transpose(2, 3) * mask

	return path

def clip_grad_value_(parameters, clip_value, norm_type=2):

	"""
	Clips gradients of the parameters to avoid exploding gradients.
	"""

	if isinstance(parameters, torch.Tensor):

		parameters = [parameters]

	parameters = list(filter(lambda p: p.grad is not None, parameters))

	norm_type = float(norm_type)

	if clip_value is not None:

		clip_value = float(clip_value)

	total_norm = 0

	for p in parameters:

		param_norm = p.grad.data.norm(norm_type)

		total_norm += param_norm.item() ** norm_type

		if clip_value is not None:

			p.grad.data.clamp_(min=-clip_value, max=clip_value)

	total_norm = total_norm ** (1. / norm_type)

	return total_norm
