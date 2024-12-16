# models.py
#
# This file contains the implementation for the whole VITS (Variational Inference Text-to-Speech) model. 
# The components in this file include the text encoder, duration predictor, generator, discriminator, and the main synthesizer model. 

import copy
import math
import torch
import commons
import modules
import attentions
import monotonic_align

from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

class StochasticDurationPredictor(nn.Module):

	"""
	A model for predicting the stochastic duration of speech segments.

	Args:
		in_channels: Number of input channels.
		filter_channels: Number of filter channels for convolutional layers.
		kernel_size: Size of the convolutional kernel.
		p_dropout: Dropout probability to prevent overfitting.
		n_flows: Number of flow layers used in the model.
		gin_channels: Conditional input channels, if any.

	Methods:
		forward: Processes the input tensor and predicts the duration of the segments.
	"""

	def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):

		super().__init__()

		filter_channels = in_channels

		self.in_channels = in_channels
		self.filter_channels = filter_channels
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.n_flows = n_flows
		self.gin_channels = gin_channels
		self.log_flow = modules.Log()
		self.flows = nn.ModuleList()
		self.flows.append(modules.ElementwiseAffine(2))

		for i in range(n_flows):

			self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
			self.flows.append(modules.Flip())

		self.post_pre = nn.Conv1d(1, filter_channels, 1)
		self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
		self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
		self.post_flows = nn.ModuleList()
		self.post_flows.append(modules.ElementwiseAffine(2))

		for i in range(4):

			self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
			self.post_flows.append(modules.Flip())

		self.pre = nn.Conv1d(in_channels, filter_channels, 1)
		self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
		self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)

		if gin_channels != 0:

			self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

	def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):

		"""
		Forward pass for the Stochastic Duration Predictor. 

		Args:
			x: Input tensor, usually a feature map from the previous network layer.
			x_mask: Mask for the input tensor (used for padding).
			w: Optional input for conditional duration prediction.
			g: Optional input for conditioning the model on additional features.
			reverse: Whether to reverse the flow direction during inference (default: False).
			noise_scale: Scaling factor for the random noise used in stochastic prediction.

		Returns:
			A tensor representing the predicted durations, log-determinants, or other relevant outputs.
		"""

		x = torch.detach(x)
		x = self.pre(x)

		if g is not None:

			g = torch.detach(g)

			x = x + self.cond(g)

		x = self.convs(x, x_mask)
		x = self.proj(x) * x_mask

		if not reverse:

			flows = self.flows

			assert w is not None

			logdet_tot_q = 0

			h_w = self.post_pre(w)
			h_w = self.post_convs(h_w, x_mask)
			h_w = self.post_proj(h_w) * x_mask

			e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask

			z_q = e_q

			for flow in self.post_flows:

				z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))

				logdet_tot_q += logdet_q

			z_u, z1 = torch.split(z_q, [1, 1], 1)

			u = torch.sigmoid(z_u) * x_mask

			z0 = (w - u) * x_mask

			logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])

			logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

			logdet_tot = 0

			z0, logdet = self.log_flow(z0, x_mask)

			logdet_tot += logdet

			z = torch.cat([z0, z1], 1)

			for flow in flows:

				z, logdet = flow(z, x_mask, g=x, reverse=reverse)

				logdet_tot = logdet_tot + logdet

			nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot

			return nll + logq  # [b]

		else:

			flows = list(reversed(self.flows))
			flows = flows[:-2] + [flows[-1]]

			z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale

			for flow in flows:

				z = flow(z, x_mask, g=x, reverse=reverse)

			z0, z1 = torch.split(z, [1, 1], 1)

			logw = z0

			return logw

class DurationPredictor(nn.Module):

	"""
	A model for predicting the duration of speech segments based on input features.

	Args:
		in_channels: Number of input channels.
		filter_channels: Number of filter channels for convolutional layers.
		kernel_size: Size of the convolutional kernel.
		p_dropout: Dropout probability for regularization.
		gin_channels: Number of input channels for conditional inputs, if any.

	Methods:
		forward: Process the input and return the predicted duration.
	"""

	def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):

		super().__init__()

		self.in_channels = in_channels
		self.filter_channels = filter_channels
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.gin_channels = gin_channels
		self.drop = nn.Dropout(p_dropout)
		self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
		self.norm_1 = modules.LayerNorm(filter_channels)
		self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
		self.norm_2 = modules.LayerNorm(filter_channels)
		self.proj = nn.Conv1d(filter_channels, 1, 1)

		if gin_channels != 0:

			self.cond = nn.Conv1d(gin_channels, in_channels, 1)

	def forward(self, x, x_mask, g=None):

		"""
		Forward pass to predict the duration of speech segments.

		Args:
			x: Input tensor representing speech features.
			x_mask: Mask to indicate valid input values.
			g: Optional tensor for additional conditioning.

		Returns:
			A tensor containing predicted durations for the segments.
		"""

		x = torch.detach(x)

		if g is not None:

			g = torch.detach(g)

			x = x + self.cond(g)

		x = self.conv_1(x * x_mask)

		x = torch.relu(x)

		x = self.norm_1(x)
		x = self.drop(x)
		x = self.conv_2(x * x_mask)

		x = torch.relu(x)

		x = self.norm_2(x)
		x = self.drop(x)
		x = self.proj(x * x_mask)

		return x * x_mask

class TextEncoder(nn.Module):

	"""
	A model to encode input text into a sequence of feature representations using an attention-based encoder.

	Args:
		n_vocab: Size of the vocabulary.
		out_channels: Number of output channels for the encoder.
		hidden_channels: Number of hidden channels.
		filter_channels: Number of filter channels in the encoder.
		n_heads: Number of attention heads.
		n_layers: Number of layers in the encoder.
		kernel_size: Size of the convolutional kernel used in the encoder.
		p_dropout: Dropout probability for regularization.

	Methods:
		forward: Process the input text and return the encoded features.
	"""

	def __init__(self,
				 n_vocab,
				 out_channels,
				 hidden_channels,
				 filter_channels,
				 n_heads,
				 n_layers,
				 kernel_size,
				 p_dropout):
		super().__init__()

		# Initialize embedding and encoder layers
		self.n_vocab = n_vocab
		self.out_channels = out_channels
		self.hidden_channels = hidden_channels
		self.filter_channels = filter_channels
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout

		self.emb = nn.Embedding(n_vocab, hidden_channels)
		nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

		self.encoder = attentions.Encoder(
			hidden_channels,
			filter_channels,
			n_heads,
			n_layers,
			kernel_size,
			p_dropout
		)

		self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

	def forward(self, x, x_lengths):

		"""
		Forward pass to process input text and generate feature representations.

		Args:
			x: Input tensor representing the text sequence (e.g., word indices).
			x_lengths: Lengths of each text sequence.

		Returns:
			Tuple of encoded features, mean, log-variance, and the input mask.
		"""

		x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
		x = torch.transpose(x, 1, -1)  # [b, h, t]

		x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

		x = self.encoder(x * x_mask, x_mask)

		stats = self.proj(x) * x_mask

		m, logs = torch.split(stats, self.out_channels, dim=1)

		return x, m, logs, x_mask

class ResidualCouplingBlock(nn.Module):

	"""
	Residual coupling block for normalizing flows. It consists of several residual coupling layers
	and flip operations to transform input data in a flow-based manner.
	
	Args:
		channels: Number of input channels.
		hidden_channels: Number of hidden channels.
		kernel_size: Size of the convolutional kernel.
		dilation_rate: Dilation rate for convolutions.
		n_layers: Number of layers in the residual coupling blocks.
		n_flows: Number of flow operations (default: 4).
		gin_channels: Number of conditioning input channels (default: 0).
	"""

	def __init__(
			self,
			channels,
			hidden_channels,
			kernel_size,
			dilation_rate,
			n_layers,
			n_flows=4,
			gin_channels=0
		):

		super().__init__()

		self.channels = channels
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.dilation_rate = dilation_rate
		self.n_layers = n_layers
		self.n_flows = n_flows
		self.gin_channels = gin_channels
		self.flows = nn.ModuleList()

		for i in range(n_flows):

			self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
			self.flows.append(modules.Flip())

	def forward(self, x, x_mask, g=None, reverse=False):

		"""
		Apply the flow-based transformation to the input tensor `x`.
		
		Args:
			x: The input tensor to be transformed.
			x_mask: Mask to indicate valid input values.
			g: Optional conditioning input (default: None).
			reverse: Whether to reverse the flow transformations (default: False).
			
		Returns:
			Transformed tensor.
		"""

		if not reverse:

			for flow in self.flows:

				x, _ = flow(x, x_mask, g=g, reverse=reverse)

		else:

			for flow in reversed(self.flows):

				x = flow(x, x_mask, g=g, reverse=reverse)

		return x

class PosteriorEncoder(nn.Module):

	"""
	Posterior encoder that transforms an input sequence `y` into a latent space representation
	by predicting the mean and log-variance parameters of a Gaussian distribution.
	
	Args:
		in_channels: Number of input channels (usually the spectrogram channels).
		out_channels: Number of output channels (usually the latent dimension).
		hidden_channels: Number of hidden channels in the convolutional layers.
		kernel_size: Size of the kernel in the convolutional layers.
		dilation_rate: Dilation rate in the convolutions.
		n_layers: Number of layers in the encoder.
		gin_channels: Number of conditioning input channels (default: 0).
	"""

	def __init__(
		self,
		in_channels,
		out_channels,
		hidden_channels,
		kernel_size,
		dilation_rate,
		n_layers,
		gin_channels=0
	):

		super().__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.dilation_rate = dilation_rate
		self.n_layers = n_layers
		self.gin_channels = gin_channels
		self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
		self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
		self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

	def forward(self, x, x_lengths, g=None):

		"""
		Encode the input sequence `y` into the latent space representation.
		
		Args:
			x: The input tensor.
			x_lengths: Lengths of the input sequences.
			g: Optional conditioning input (default: None).
			
		Returns:
			z: Latent representation.
			m: Mean of the posterior distribution.
			logs: Log variance of the posterior distribution.
			x_mask: Mask to handle variable-length sequences.
		"""

		x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

		x = self.pre(x) * x_mask
		x = self.enc(x, x_mask, g=g)

		stats = self.proj(x) * x_mask

		m, logs = torch.split(stats, self.out_channels, dim=1)

		z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

		return z, m, logs, x_mask

class Generator(torch.nn.Module):

	"""
	Generator for synthesizing the output from latent variables, using upsampling and residual blocks.
	
	Args:
		initial_channel: Number of initial input channels.
		resblock: Type of residual block to use ('1' or '2').
		resblock_kernel_sizes: Kernel sizes for the residual blocks.
		resblock_dilation_sizes: Dilation sizes for the residual blocks.
		upsample_rates: Upsampling rates for each stage.
		upsample_initial_channel: Initial number of channels for upsampling.
		upsample_kernel_sizes: Kernel sizes for upsampling.
		gin_channels: Number of conditioning input channels (default: 0).
	"""

	def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):

		super(Generator, self).__init__()

		self.num_kernels = len(resblock_kernel_sizes)
		self.num_upsamples = len(upsample_rates)
		self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

		resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

		self.ups = nn.ModuleList()

		for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):

			self.ups.append(weight_norm(

				ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),k, u, padding=(k-u)//2)))

		self.resblocks = nn.ModuleList()

		for i in range(len(self.ups)):

			ch = upsample_initial_channel//(2**(i+1))

			for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):

				self.resblocks.append(resblock(ch, k, d))

		self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
		self.ups.apply(init_weights)

		if gin_channels != 0:

			self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

	def forward(self, x, g=None):

		"""
		Generate output from latent representation `x`.
		
		Args:
			x: The input latent tensor.
			g: Optional conditioning input (default: None).
			
		Returns:
			Generated tensor.
		"""

		x = self.conv_pre(x)

		if g is not None:

			x = x + self.cond(g)

		for i in range(self.num_upsamples):

			x = F.leaky_relu(x, modules.LRELU_SLOPE)
			x = self.ups[i](x)

			xs = None

			for j in range(self.num_kernels):

				if xs is None:

					xs = self.resblocks[i*self.num_kernels+j](x)

				else:

					xs += self.resblocks[i*self.num_kernels+j](x)

			x = xs / self.num_kernels

		x = F.leaky_relu(x)
		x = self.conv_post(x)
		x = torch.tanh(x)

		return x

	def remove_weight_norm(self):

		"""
		Remove weight normalization from all layers.
		"""

		print('Removing weight norm...')

		for l in self.ups:

			remove_weight_norm(l)

		for l in self.resblocks:

			l.remove_weight_norm()

class DiscriminatorP(torch.nn.Module):

	"""
	Periodic Discriminator for multi-scale period discrimination. This class processes input 
	spectrograms using a series of 2D convolutions to assess the quality of generated samples 
	by comparing their periodic structure to real samples.

	Args:
		period: The period for the input spectrogram (used for padding).
		kernel_size: Size of the convolution kernel (default: 5).
		stride: Stride for convolutions (default: 3).
		use_spectral_norm: If True, applies spectral normalization (default: False).
	"""

	def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):

		super(DiscriminatorP, self).__init__()

		self.period = period
		self.use_spectral_norm = use_spectral_norm

		norm_f = weight_norm if use_spectral_norm == False else spectral_norm

		self.convs = nn.ModuleList([
			norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
			norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
			norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
			norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
			norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
		])

		self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

	def forward(self, x):

		"""
		Forward pass through the discriminator.
		
		Args:
			x: The input tensor (e.g., a spectrogram) to be processed.
		
		Returns:
			x: The output of the discriminator after passing through convolutions and activation.
			fmap: Feature maps from intermediate layers.
		"""

		fmap = []

		b, c, t = x.shape

		if t % self.period != 0:

			n_pad = self.period - (t % self.period)

			x = F.pad(x, (0, n_pad), "reflect")

			t = t + n_pad

		x = x.view(b, c, t // self.period, self.period)

		for l in self.convs:

			x = l(x)
			x = F.leaky_relu(x, modules.LRELU_SLOPE)

			fmap.append(x)

		x = self.conv_post(x)

		fmap.append(x)

		x = torch.flatten(x, 1, -1)

		return x, fmap

class DiscriminatorS(torch.nn.Module):

	"""
	Simple 1D Discriminator for assessing the quality of generated sequences by comparing
	the spectrograms of real and generated audio.

	Args:
		use_spectral_norm: If True, applies spectral normalization (default: False).
	"""

	def __init__(self, use_spectral_norm=False):

		super(DiscriminatorS, self).__init__()

		norm_f = weight_norm if use_spectral_norm == False else spectral_norm

		self.convs = nn.ModuleList([
			norm_f(Conv1d(1, 16, 15, 1, padding=7)),
			norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
			norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
			norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
			norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
			norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
		])

		self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

	def forward(self, x):

		"""
		Forward pass through the discriminator.
		
		Args:
			x: The input tensor (e.g., a spectrogram) to be processed.
		
		Returns:
			x: The output of the discriminator after passing through convolutions and activation.
			fmap: Feature maps from intermediate layers.
		"""

		fmap = []

		for l in self.convs:

			x = l(x)
			x = F.leaky_relu(x, modules.LRELU_SLOPE)

			fmap.append(x)

		x = self.conv_post(x)

		fmap.append(x)

		x = torch.flatten(x, 1, -1)

		return x, fmap

class MultiPeriodDiscriminator(torch.nn.Module):

	"""
	Multi-Period Discriminator that combines multiple discriminators working at different 
	periods to capture various scales of periodic structures in the audio signal.
	
	Args:
		use_spectral_norm: If True, applies spectral normalization (default: False).
	"""

	def __init__(self, use_spectral_norm=False):

		super(MultiPeriodDiscriminator, self).__init__()

		periods = [2,3,5,7,11]

		discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
		discs += [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]

		self.discriminators = nn.ModuleList(discs)

	def forward(self, y, y_hat):

		"""
		Forward pass through the multi-period discriminator.
		
		Args:
			y: The real audio tensor.
			y_hat: The generated audio tensor.
		
		Returns:
			y_d_rs: List of real discriminators' outputs.
			y_d_gs: List of generated discriminators' outputs.
			fmap_rs: Feature maps from real discriminators.
			fmap_gs: Feature maps from generated discriminators.
		"""

		y_d_rs = []
		y_d_gs = []

		fmap_rs = []
		fmap_gs = []

		for i, d in enumerate(self.discriminators):

			y_d_r, fmap_r = d(y)
			y_d_g, fmap_g = d(y_hat)
			y_d_rs.append(y_d_r)
			y_d_gs.append(y_d_g)

			fmap_rs.append(fmap_r)
			fmap_gs.append(fmap_g)

		return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class SynthesizerTrn(nn.Module):

	"""
	Synthesizer for Training: A neural network model for text-to-speech synthesis.
	It encodes input text, predicts durations, and generates spectrograms.
	"""

	def __init__(
		self, 
		n_vocab,
		spec_channels,
		segment_size,
		inter_channels,
		hidden_channels,
		filter_channels,
		n_heads,
		n_layers,
		kernel_size,
		p_dropout,
		resblock,
		resblock_kernel_sizes,
		resblock_dilation_sizes,
		upsample_rates,
		upsample_initial_channel,
		upsample_kernel_sizes,
		n_speakers=0,
		gin_channels=0,
		use_sdp=True,
		**kwargs
	):

		super().__init__()

		self.n_vocab = n_vocab
		self.spec_channels = spec_channels
		self.segment_size = segment_size
		self.inter_channels = inter_channels
		self.hidden_channels = hidden_channels
		self.filter_channels = filter_channels
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.resblock = resblock
		self.resblock_kernel_sizes = resblock_kernel_sizes
		self.resblock_dilation_sizes = resblock_dilation_sizes
		self.upsample_rates = upsample_rates
		self.upsample_initial_channel = upsample_initial_channel
		self.upsample_kernel_sizes = upsample_kernel_sizes
		self.n_speakers = n_speakers
		self.gin_channels = gin_channels
		self.use_sdp = use_sdp

		self.enc_p = TextEncoder(
			n_vocab,
			inter_channels,
			hidden_channels,
			filter_channels,
			n_heads,
			n_layers,
			kernel_size,
			p_dropout
		)

		self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,  upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
		self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
		self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

		if use_sdp:

			self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)

		else:

			self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

		if n_speakers > 1:
			self.emb_g = nn.Embedding(n_speakers, gin_channels)

	def forward(self, x, x_lengths, y, y_lengths, sid=None):

		"""
		Forward pass for the model during training.
		Encodes input text and spectrograms, predicts durations, and generates a spectrogram.
		"""

		x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

		if self.n_speakers > 0:

			g = self.emb_g(sid).unsqueeze(-1)

		else:

			g = None

		z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
		
		z_p = self.flow(z, y_mask, g=g)

		with torch.no_grad():

			s_p_sq_r = torch.exp(-2 * logs_p)

			neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
			neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
			neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
			neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)

			neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

			attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)

			attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

		w = attn.sum(2)

		if self.use_sdp:

			l_length = self.dp(x, x_mask, w, g=g)
			l_length = l_length / torch.sum(x_mask)

		else:

			logw_ = torch.log(w + 1e-6) * x_mask
			logw = self.dp(x, x_mask, g=g)

			l_length = torch.sum((logw - logw_)**2, [1, 2]) / torch.sum(x_mask)

		m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)

		logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

		z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)

		o = self.dec(z_slice, g=g)

		return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

	def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):

		"""
		Inference pass for generating speech from input text.
		"""

		x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

		if self.n_speakers > 0:

			g = self.emb_g(sid).unsqueeze(-1)

		else:

			g = None

		if self.use_sdp:

			logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)

		else:

			logw = self.dp(x, x_mask, g=g)

		w = torch.exp(logw) * x_mask * length_scale

		w_ceil = torch.ceil(w)

		y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()

		y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)

		attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)

		attn = commons.generate_path(w_ceil, attn_mask)

		m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)

		logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

		z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

		z = self.flow(z_p, y_mask, g=g, reverse=True)

		o = self.dec((z * y_mask)[:,:,:max_len], g=g)

		return o, attn, y_mask, (z, z_p, m_p, logs_p)

	def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):

		"""
		Voice conversion: converts the voice of one speaker to another.
		"""

		assert self.n_speakers > 0, "n_speakers have to be larger than 0."

		g_src = self.emb_g(sid_src).unsqueeze(-1)
		g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)

		z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)

		z_p = self.flow(z, y_mask, g=g_src)

		z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)

		o_hat = self.dec(z_hat * y_mask, g=g_tgt)

		return o_hat, y_mask, (z, z_p, z_hat)
