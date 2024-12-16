# utils.py
#
# This utility file provides various functions for loading and saving model checkpoints,
# visualizing data (like spectrograms and alignments), managing configuration files,
# handling logging, and interacting with file systems.

import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
import torch

from scipy.io.wavfile import read

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

def load_checkpoint(checkpoint_path, model, optimizer=None):

	"""
	Load model and optimizer state from a checkpoint.
	
	Args:
		checkpoint_path (str): Path to the checkpoint file.
		model (torch.nn.Module): The model to load the state into.
		optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.
	
	Returns:
		tuple: The model, optimizer, learning rate, and iteration from the checkpoint.
	"""

	assert os.path.isfile(checkpoint_path)

	checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

	iteration = checkpoint_dict['iteration']

	learning_rate = checkpoint_dict['learning_rate']

	if optimizer is not None:

		optimizer.load_state_dict(checkpoint_dict['optimizer'])

	saved_state_dict = checkpoint_dict['model']

	if hasattr(model, 'module'):

		state_dict = model.module.state_dict()

	else:

		state_dict = model.state_dict()

	new_state_dict = {}

	for k, v in state_dict.items():

		try:

			new_state_dict[k] = saved_state_dict[k]

		except:

			logger.info("%s is not in the checkpoint" % k)

			new_state_dict[k] = v

	if hasattr(model, 'module'):

		model.module.load_state_dict(new_state_dict)

	else:

		model.load_state_dict(new_state_dict)

	logger.info("Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration))

	return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):

	"""
	Save model and optimizer state to a checkpoint file.
	
	Args:
		model (torch.nn.Module): The model whose state will be saved.
		optimizer (torch.optim.Optimizer): The optimizer whose state will be saved.
		learning_rate (float): The learning rate to save.
		iteration (int): The iteration number to save.
		checkpoint_path (str): The path to save the checkpoint file.
	"""

	logger.info("Saving model and optimizer state at iteration {} to {}".format(iteration, checkpoint_path))

	if hasattr(model, 'module'):

		state_dict = model.module.state_dict()

	else:

		state_dict = model.state_dict()

	torch.save({
		'model': state_dict,
		'iteration': iteration,
		'optimizer': optimizer.state_dict(),
		'learning_rate': learning_rate
	}, checkpoint_path)

def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):

	"""
	Log various types of data (scalars, histograms, images, audios) to TensorBoard.
	
	Args:
		writer (SummaryWriter): The TensorBoard writer instance.
		global_step (int): The global step (iteration) for the current logging.
		scalars (dict): A dictionary of scalar values to log.
		histograms (dict): A dictionary of histogram values to log.
		images (dict): A dictionary of image data to log.
		audios (dict): A dictionary of audio data to log.
		audio_sampling_rate (int, optional): The sampling rate of audio data.
	"""

	for k, v in scalars.items():

		writer.add_scalar(k, v, global_step)

	for k, v in histograms.items():

		writer.add_histogram(k, v, global_step)

	for k, v in images.items():

		writer.add_image(k, v, global_step, dataformats='HWC')

	for k, v in audios.items():

		writer.add_audio(k, v, global_step, audio_sampling_rate)

def latest_checkpoint_path(dir_path, regex="G_*.pth"):

	"""
	Find the latest checkpoint file in a given directory.
	
	Args:
		dir_path (str): The directory to search for checkpoint files.
		regex (str, optional): The regex pattern to match checkpoint filenames.
	
	Returns:
		str: The path of the latest checkpoint file.
	"""

	f_list = glob.glob(os.path.join(dir_path, regex))

	f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

	x = f_list[-1]

	print(x)

	return x

def plot_spectrogram_to_numpy(spectrogram):

	"""
	Convert a spectrogram into a numpy array for visualization.
	
	Args:
		spectrogram (numpy.ndarray): The spectrogram to visualize.
	
	Returns:
		numpy.ndarray: The RGB numpy array of the plot.
	"""

	global MATPLOTLIB_FLAG

	if not MATPLOTLIB_FLAG:

		import matplotlib

		matplotlib.use("Agg")

		MATPLOTLIB_FLAG = True

		mpl_logger = logging.getLogger('matplotlib')
		mpl_logger.setLevel(logging.WARNING)

	import matplotlib.pylab as plt

	import numpy as np
	
	fig, ax = plt.subplots(figsize=(10, 2))

	im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')

	plt.colorbar(im, ax=ax)
	plt.xlabel("Frames")
	plt.ylabel("Channels")
	plt.tight_layout()

	fig.canvas.draw()

	data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

	plt.close()

	return data

def plot_alignment_to_numpy(alignment, info=None):

	"""
	Convert an alignment matrix to a numpy array for visualization.
	
	Args:
		alignment (numpy.ndarray): The alignment matrix to visualize.
		info (str, optional): Additional information to display on the plot.
	
	Returns:
		numpy.ndarray: The RGB numpy array of the plot.
	"""

	global MATPLOTLIB_FLAG

	if not MATPLOTLIB_FLAG:

		import matplotlib

		matplotlib.use("Agg")

		MATPLOTLIB_FLAG = True

		mpl_logger = logging.getLogger('matplotlib')
		mpl_logger.setLevel(logging.WARNING)

	import matplotlib.pylab as plt
	import numpy as np

	fig, ax = plt.subplots(figsize=(6, 4))

	im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower', interpolation='none')

	fig.colorbar(im, ax=ax)

	xlabel = 'Decoder timestep'

	if info is not None:

		xlabel += '\n\n' + info

	plt.xlabel(xlabel)
	plt.ylabel('Encoder timestep')
	plt.tight_layout()

	fig.canvas.draw()

	data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

	plt.close()

	return data

def load_wav_to_torch(full_path):

	"""
	Load a WAV file and return the audio data as a PyTorch tensor.
	
	Args:
		full_path (str): The path to the WAV file.
	
	Returns:
		tuple: The audio data as a PyTorch tensor and the sampling rate.
	"""

	sampling_rate, data = read(full_path)

	return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(filename, split="|"):

	"""
	Load file paths and associated text from a given file.
	
	Args:
		filename (str): The path to the file containing the data.
		split (str, optional): The delimiter used to split the file paths and text.
	
	Returns:
		list: A list of file paths and associated text.
	"""

	with open(filename, encoding='utf-8') as f:

		filepaths_and_text = [line.strip().split(split) for line in f]

	return filepaths_and_text

def get_hparams(init=True):

	"""
	Parse the hyperparameters from a configuration file and return them.
	
	Args:
		init (bool, optional): Whether to initialize the hyperparameters from the config file.
	
	Returns:
		HParams: An instance of the HParams class containing the hyperparameters.
	"""

	parser = argparse.ArgumentParser()

	parser.add_argument('-c', '--config', type=str, default="./configs/base.json", help='JSON file for configuration')
	parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
	
	args = parser.parse_args()

	model_dir = os.path.join("./logs", args.model)

	if not os.path.exists(model_dir):

		os.makedirs(model_dir)

	config_path = args.config

	config_save_path = os.path.join(model_dir, "config.json")

	if init:

		with open(config_path, "r") as f:

			data = f.read()

		with open(config_save_path, "w") as f:

			f.write(data)

	else:

		with open(config_save_path, "r") as f:

			data = f.read()

	config = json.loads(data)
	
	hparams = HParams(**config)

	hparams.model_dir = model_dir

	return hparams

def get_hparams_from_dir(model_dir):

	"""
	Load hyperparameters from a model directory.
	
	Args:
		model_dir (str): The directory containing the configuration file.
	
	Returns:
		HParams: An instance of the HParams class containing the hyperparameters.
	"""

	config_save_path = os.path.join(model_dir, "config.json")

	with open(config_save_path, "r") as f:

		data = f.read()

	config = json.loads(data)

	hparams = HParams(**config)

	hparams.model_dir = model_dir

	return hparams

def get_hparams_from_file(config_path):

	"""
	Load hyperparameters from a configuration file.
	
	Args:
		config_path (str): The path to the configuration file.
	
	Returns:
		HParams: An instance of the HParams class containing the hyperparameters.
	"""

	with open(config_path, "r") as f:

		data = f.read()

	config = json.loads(data)

	hparams = HParams(**config)

	return hparams

def check_git_hash(model_dir):

	"""
	Verify that the current Git commit hash matches the saved one in the model directory.
	
	Args:
		model_dir (str): The model directory where the Git hash is stored.
	"""

	source_dir = os.path.dirname(os.path.realpath(__file__))

	if not os.path.exists(os.path.join(source_dir, ".git")):

		logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(source_dir))

		return

	cur_hash = subprocess.getoutput("git rev-parse HEAD")

	path = os.path.join(model_dir, "githash")

	if os.path.exists(path):

		saved_hash = open(path).read()

		if saved_hash != cur_hash:

			logger.warn("git hash values are different. {}(saved) != {}(current)".format(saved_hash[:8], cur_hash[:8]))

	else:

		open(path, "w").write(cur_hash)

def get_logger(model_dir, filename="train.log"):

	"""
	Initialize a logger for tracking training progress.
	
	Args:
		model_dir (str): The model directory where the log file will be saved.
		filename (str, optional): The name of the log file.
	
	Returns:
		logger: The logger instance.
	"""

	global logger

	logger = logging.getLogger(os.path.basename(model_dir))

	logger.setLevel(logging.DEBUG)
	
	formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

	if not os.path.exists(model_dir):

		os.makedirs(model_dir)

	h = logging.FileHandler(os.path.join(model_dir, filename))

	h.setLevel(logging.DEBUG)
	h.setFormatter(formatter)

	logger.addHandler(h)

	return logger

class HParams():

	"""
	A class for storing hyperparameters.
	"""

	def __init__(self, **kwargs):

		for k, v in kwargs.items():

			if type(v) == dict:

				v = HParams(**v)

			self[k] = v
	
	def keys(self):

		return self.__dict__.keys()

	def items(self):

		return self.__dict__.items()

	def values(self):

		return self.__dict__.values()

	def __len__(self):

		return len(self.__dict__)

	def __getitem__(self, key):

		return getattr(self, key)

	def __setitem__(self, key, value):

		return setattr(self, key, value)

	def __contains__(self, key):

		return key in self.__dict__

	def __repr__(self):

		return self.__dict__.__repr__()
