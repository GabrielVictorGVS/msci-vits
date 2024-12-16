# losses.py
#
# This file contains the implementation of various loss functions used in the VITS (Variational Inference Text-to-Speech) model:
# - feature_loss: Measures the difference between feature maps of real and generated speech.
# - discriminator_loss: Computes the loss for the discriminator, which distinguishes real from generated samples.
# - generator_loss: Computes the loss for the generator, encouraging it to produce high-quality samples.
# - kl_loss: Computes the Kullback-Leibler divergence loss between two distributions, helping to regularize the latent variables.
#
# These loss functions are crucial for training the model to generate realistic speech.

import torch 
import commons

from torch.nn import functional as F

def feature_loss(fmap_r, fmap_g):

	"""
	Computes the feature loss between real and generated feature maps.

	Args:
	- fmap_r: Feature maps of the real samples (list of tensors)
	- fmap_g: Feature maps of the generated samples (list of tensors)

	Returns:
	- loss: The computed feature loss, encouraging similarity between real and generated features
	"""

	loss = 0

	for dr, dg in zip(fmap_r, fmap_g):

		for rl, gl in zip(dr, dg):

			rl = rl.float().detach()
			gl = gl.float()

			loss += torch.mean(torch.abs(rl - gl))

	return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):

	"""
	Computes the discriminator loss, which distinguishes between real and generated samples.

	Args:
	- disc_real_outputs: Outputs of the discriminator for real samples (list of tensors)
	- disc_generated_outputs: Outputs of the discriminator for generated samples (list of tensors)

	Returns:
	- loss: Total loss for the discriminator
	- r_losses: Losses computed for real samples
	- g_losses: Losses computed for generated samples
	"""

	loss = 0

	r_losses = []
	g_losses = []

	for dr, dg in zip(disc_real_outputs, disc_generated_outputs):

		dr = dr.float()
		dg = dg.float()

		r_loss = torch.mean((1 - dr)**2)
		g_loss = torch.mean(dg**2)

		loss += (r_loss + g_loss)

		r_losses.append(r_loss.item())
		g_losses.append(g_loss.item())

	return loss, r_losses, g_losses

def generator_loss(disc_outputs):

	"""
	Computes the generator loss, encouraging the generator to produce realistic samples.

	Args:
	- disc_outputs: Discriminator outputs for the generated samples (list of tensors)

	Returns:
	- loss: Total generator loss
	- gen_losses: Individual losses computed for each generated sample
	"""

	loss = 0

	gen_losses = []

	for dg in disc_outputs:

		dg = dg.float()

		l = torch.mean((1 - dg)**2)

		gen_losses.append(l)

		loss += l

	return loss, gen_losses

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):

	"""
	Computes the Kullback-Leibler (KL) divergence loss between two distributions for regularizing latent variables.

	Args:
	- z_p: Latent variables from the posterior distribution [b, h, t_t]
	- logs_q: Logarithm of the variance of the posterior distribution [b, h, t_t]
	- m_p: Mean of the prior distribution [b, h, t_t]
	- logs_p: Logarithm of the variance of the prior distribution [b, h, t_t]
	- z_mask: Mask indicating valid positions in the input sequence [b, 1, t_t]

	Returns:
	- l: The computed KL divergence loss
	"""

	z_p = z_p.float()

	logs_q = logs_q.float()

	m_p = m_p.float()

	logs_p = logs_p.float()

	z_mask = z_mask.float()

	kl = logs_p - logs_q - 0.5
	kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
	kl = torch.sum(kl * z_mask)

	l = kl / torch.sum(z_mask)

	return l
