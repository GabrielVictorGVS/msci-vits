# mel_processing.py
#
# This file contains utility functions for mel spectrogram processing using PyTorch.
# These include functions for dynamic range compression/decompression, spectral normalization, and mel spectrogram extraction.

import math
import os
import random
import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import librosa
import librosa.util as librosa_util

from torch import nn
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):

	"""
	Perform dynamic range compression on the input tensor `x` using a logarithmic function.
	
	Args:
		x: Tensor to compress.
		C: Compression factor (default: 1).
		clip_val: Minimum value to clip the tensor before applying log (default: 1e-5).
		
	Returns:
		Compressed tensor.
	"""

	return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):

	"""
	Perform dynamic range decompression on the input tensor `x` by exponentiating.
	
	Args:
		x: Tensor to decompress.
		C: Compression factor used during compression (default: 1).
		
	Returns:
		Decompressed tensor.
	"""

	return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):

	"""
	Apply spectral normalization by compressing the range of magnitudes.
	
	Args:
		magnitudes: Input tensor to normalize.
		
	Returns:
		Normalized tensor.
	"""

	output = dynamic_range_compression_torch(magnitudes)

	return output

def spectral_de_normalize_torch(magnitudes):

	"""
	Apply spectral de-normalization (inverse of normalization) to the magnitudes.
	
	Args:
		magnitudes: Input tensor to denormalize.
		
	Returns:
		De-normalized tensor.
	"""

	output = dynamic_range_decompression_torch(magnitudes)

	return output

mel_basis = {}

hann_window = {}

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):

	"""
	Generate the spectrogram of the input waveform `y` using the Short-Time Fourier Transform (STFT).
	
	Args:
		y: Input tensor representing the waveform.
		n_fft: Size of FFT (default: 1024).
		sampling_rate: The sampling rate of the waveform.
		hop_size: Number of samples between successive frames.
		win_size: Size of the window used for STFT.
		center: Whether to center the input signal before applying the FFT (default: False).
		
	Returns:
		Spectrogram of the input tensor.
	"""

	if torch.min(y) < -1.:

		print('min value is ', torch.min(y))

	if torch.max(y) > 1.:

		print('max value is ', torch.max(y))

	global hann_window

	dtype_device = str(y.dtype) + '_' + str(y.device)

	wnsize_dtype_device = str(win_size) + '_' + dtype_device

	if wnsize_dtype_device not in hann_window:

		hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

	y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
	y = y.squeeze(1)

	spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device], center=center, pad_mode='reflect', normalized=False, onesided=True)
	spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

	return spec

def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):

	"""
	Convert spectrogram to mel spectrogram using a precomputed mel basis.
	
	Args:
		spec: Spectrogram tensor to convert.
		n_fft: Size of FFT used to compute the spectrogram.
		num_mels: Number of mel bins.
		sampling_rate: Sampling rate of the waveform.
		fmin: Minimum frequency for mel filterbank.
		fmax: Maximum frequency for mel filterbank.
		
	Returns:
		Mel spectrogram tensor.
	"""

	global mel_basis

	dtype_device = str(spec.dtype) + '_' + str(spec.device)

	fmax_dtype_device = str(fmax) + '_' + dtype_device

	if fmax_dtype_device not in mel_basis:

		mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
		mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)

	spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
	spec = spectral_normalize_torch(spec)

	return spec

def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):

	"""
	Generate a mel spectrogram from the input waveform `y`.
	
	Args:
		y: Input tensor representing the waveform.
		n_fft: Size of FFT (default: 1024).
		num_mels: Number of mel bins (default: 80).
		sampling_rate: Sampling rate of the waveform.
		hop_size: Number of samples between successive frames.
		win_size: Size of the window used for STFT.
		fmin: Minimum frequency for mel filterbank.
		fmax: Maximum frequency for mel filterbank.
		center: Whether to center the input signal before applying the FFT (default: False).
		
	Returns:
		Mel spectrogram tensor.
	"""

	if torch.min(y) < -1.:

		print('min value is ', torch.min(y))

	if torch.max(y) > 1.:

		print('max value is ', torch.max(y))

	global mel_basis, hann_window

	dtype_device = str(y.dtype) + '_' + str(y.device)

	fmax_dtype_device = str(fmax) + '_' + dtype_device

	wnsize_dtype_device = str(win_size) + '_' + dtype_device

	if fmax_dtype_device not in mel_basis:

		mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
		mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)

	if wnsize_dtype_device not in hann_window:

		hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

	y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
	y = y.squeeze(1)

	spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device], center=center, pad_mode='reflect', normalized=False, onesided=True)
	spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
	spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
	spec = spectral_normalize_torch(spec)

	return spec
