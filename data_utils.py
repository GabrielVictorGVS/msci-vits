# data_utils.py
#
# This file contains utility functions and classes for loading, processing, and batching 
# data used in the Text-to-Speech (TTS) model. It includes functions for loading audio-text pairs, 
# normalizing text, computing spectrograms, and batching data with zero-padding.

import time
import os
import random
import numpy as np
import torch
import torch.utils.data
import commons

from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence

class TextAudioLoader(torch.utils.data.Dataset):

	"""
	Load audio and text pairs, normalize text, and compute spectrograms.

	Attributes:
		audiopaths_and_text (list): List of file paths and corresponding text.
		text_cleaners (list): List of text cleaning methods.
		max_wav_value (float): Maximum value for audio normalization.
		sampling_rate (int): Sampling rate for audio files.
		filter_length (int): Length of the filter for spectrogram computation.
		hop_length (int): Hop length for spectrogram computation.
		win_length (int): Window length for spectrogram computation.
		add_blank (bool): Whether to add a blank token between each character in the text.
		min_text_len (int): Minimum length of the text sequence.
		max_text_len (int): Maximum length of the text sequence.
	"""

	def __init__(self, audiopaths_and_text, hparams):

		self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
		self.text_cleaners = hparams.text_cleaners
		self.max_wav_value = hparams.max_wav_value
		self.sampling_rate = hparams.sampling_rate
		self.filter_length = hparams.filter_length
		self.hop_length = hparams.hop_length
		self.win_length = hparams.win_length
		self.add_blank = hparams.add_blank
		self.cleaned_text = getattr(hparams, "cleaned_text", False)
		self.min_text_len = getattr(hparams, "min_text_len", 1)
		self.max_text_len = getattr(hparams, "max_text_len", 190)

		random.seed(1234)
		random.shuffle(self.audiopaths_and_text)

		self._filter()

	def _filter(self):

		"""
		Filter text & store spectrogram lengths for bucketing.
		Filters the dataset based on text length and computes the lengths of the spectrograms.
		"""

		audiopaths_and_text_new = []

		lengths = []

		for audiopath, text in self.audiopaths_and_text:

			if self.min_text_len <= len(text) and len(text) <= self.max_text_len:

				audiopaths_and_text_new.append([audiopath, text])

				lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))

		self.audiopaths_and_text = audiopaths_and_text_new
		self.lengths = lengths

	def get_audio_text_pair(self, audiopath_and_text):

		"""
		Retrieve the text and spectrogram pair for a given audio-text path.

		Args:
			audiopath_and_text (tuple): A tuple containing the audio file path and the text.

		Returns:
			tuple: A tuple of (text, spectrogram, waveform) for the audio-text pair.
		"""

		audiopath, text = audiopath_and_text[0], audiopath_and_text[1]

		text = self.get_text(text)

		spec, wav = self.get_audio(audiopath)

		return (text, spec, wav)

	def get_audio(self, filename):

		"""
		Load audio, check for sampling rate mismatch, normalize, and compute spectrogram.

		Args:
			filename (str): The path to the audio file.

		Returns:
			tuple: A tuple of (spectrogram, normalized audio) for the given file.
		"""

		audio, sampling_rate = load_wav_to_torch(filename)

		if sampling_rate != self.sampling_rate:

			raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")

		audio_norm = audio / self.max_wav_value
		audio_norm = audio_norm.unsqueeze(0)

		spec_filename = filename.replace(".wav", ".spec.pt")

		if os.path.exists(spec_filename):

			spec = torch.load(spec_filename)

		else:

			spec = spectrogram_torch(audio_norm, self.filter_length, self.sampling_rate, self.hop_length, self.win_length, center=False)
			spec = torch.squeeze(spec, 0)

			torch.save(spec, spec_filename)

		return spec, audio_norm

	def get_text(self, text):

		"""
		Normalize text based on the selected cleaning method.

		Args:
			text (str): The raw text to be normalized.

		Returns:
			torch.LongTensor: The normalized text as a tensor of integers.
		"""

		if self.cleaned_text:

			text_norm = cleaned_text_to_sequence(text)

		else:

			text_norm = text_to_sequence(text, self.text_cleaners)

		if self.add_blank:

			text_norm = commons.intersperse(text_norm, 0)

		text_norm = torch.LongTensor(text_norm)

		return text_norm

	def __getitem__(self, index):

		"""
		Get the audio-text pair for a given index in the dataset.

		Args:
			index (int): The index of the data sample.

		Returns:
			tuple: A tuple containing the text, spectrogram, and waveform.
		"""

		return self.get_audio_text_pair(self.audiopaths_and_text[index])

	def __len__(self):

		"""
		Get the total number of samples in the dataset.

		Returns:
			int: The number of samples.
		"""

		return len(self.audiopaths_and_text)

class TextAudioCollate():

	"""
	Zero-pads model inputs and targets to ensure all sequences in the batch are the same length.

	Attributes:
		return_ids (bool): Whether to return the indices of the samples in the original order.
	"""

	def __init__(self, return_ids=False):

		self.return_ids = return_ids

	def __call__(self, batch):

		"""
		Collate a batch of data from normalized text and audio.

		Args:
			batch (list): A list of tuples where each tuple contains text, spectrogram, and waveform.

		Returns:
			tuple: A tuple containing the padded text, spectrograms, waveforms, and their respective lengths.
		"""

		_, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True)

		max_text_len = max([len(x[0]) for x in batch])
		max_spec_len = max([x[1].size(1) for x in batch])
		max_wav_len = max([x[2].size(1) for x in batch])

		text_lengths = torch.LongTensor(len(batch))
		spec_lengths = torch.LongTensor(len(batch))
		wav_lengths = torch.LongTensor(len(batch))

		text_padded = torch.LongTensor(len(batch), max_text_len)
		spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
		wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

		text_padded.zero_()
		spec_padded.zero_()
		wav_padded.zero_()

		for i in range(len(ids_sorted_decreasing)):

			row = batch[ids_sorted_decreasing[i]]

			text = row[0]
			text_padded[i, :text.size(0)] = text
			text_lengths[i] = text.size(0)

			spec = row[1]
			spec_padded[i, :, :spec.size(1)] = spec
			spec_lengths[i] = spec.size(1)

			wav = row[2]
			wav_padded[i, :, :wav.size(1)] = wav
			wav_lengths[i] = wav.size(1)

		if self.return_ids:

			return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing

		return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths

class TextAudioSpeakerLoader(torch.utils.data.Dataset):

	"""
	Load audio, speaker_id, and text pairs for multi-speaker TTS models.

	Attributes:
		audiopaths_sid_text (list): List of file paths, speaker IDs, and corresponding text.
		text_cleaners (list): List of text cleaning methods.
		max_wav_value (float): Maximum value for audio normalization.
		sampling_rate (int): Sampling rate for audio files.
		filter_length (int): Length of the filter for spectrogram computation.
		hop_length (int): Hop length for spectrogram computation.
		win_length (int): Window length for spectrogram computation.
		add_blank (bool): Whether to add a blank token between each character in the text.
		min_text_len (int): Minimum length of the text sequence.
		max_text_len (int): Maximum length of the text sequence.
	"""

	def __init__(self, audiopaths_sid_text, hparams):

		self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
		self.text_cleaners = hparams.text_cleaners
		self.max_wav_value = hparams.max_wav_value
		self.sampling_rate = hparams.sampling_rate
		self.filter_length = hparams.filter_length
		self.hop_length = hparams.hop_length
		self.win_length = hparams.win_length
		self.add_blank = hparams.add_blank
		self.cleaned_text = getattr(hparams, "cleaned_text", False)
		self.min_text_len = getattr(hparams, "min_text_len", 1)
		self.max_text_len = getattr(hparams, "max_text_len", 190)

		random.seed(1234)
		random.shuffle(self.audiopaths_sid_text)

		self._filter()

	def _filter(self):

		"""
		Filter the dataset based on text length and compute the lengths of the spectrograms.
		"""

		audiopaths_sid_text_new = []

		lengths = []

		for audiopath, speaker_id, text in self.audiopaths_sid_text:

			if self.min_text_len <= len(text) and len(text) <= self.max_text_len:

				audiopaths_sid_text_new.append([audiopath, speaker_id, text])

				lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))

		self.audiopaths_sid_text = audiopaths_sid_text_new
		self.lengths = lengths

	def get_audio_text_speaker_pair(self, audiopath_sid_text):

		"""
		Retrieve the text, spectrogram, and speaker ID pair for a given audio-text-speaker path.

		Args:
			audiopath_sid_text (tuple): A tuple containing the audio file path, speaker ID, and the text.

		Returns:
			tuple: A tuple of (text, spectrogram, waveform, speaker ID) for the audio-text-speaker pair.
		"""

		audiopath, speaker_id, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]

		text = self.get_text(text)

		spec, wav = self.get_audio(audiopath)

		return (text, spec, wav, speaker_id)

	def __getitem__(self, index):

		"""
		Get the audio-text-speaker pair for a given index in the dataset.

		Args:
			index (int): The index of the data sample.

		Returns:
			tuple: A tuple containing the text, spectrogram, waveform, and speaker ID.
		"""

		return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

	def __len__(self):

		"""
		Get the total number of samples in the dataset.

		Returns:
			int: The number of samples.
		"""

		return len(self.audiopaths_sid_text)
