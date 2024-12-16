# preprocess.py
#
# This script preprocesses text data from given file lists by cleaning the text according to specified cleaners.
# It reads the raw text files, cleans the text, updates the vocabulary with new symbols, and saves the cleaned data to new file lists.
# The script is useful for preparing text data for training a speech synthesis model, such as with the MSCI-PT dataset.

import argparse
import text

from utils import load_filepaths_and_text

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--out_extension", default="clean", help="The extension for output file lists (default: 'clean')")
	parser.add_argument("--text_index", default=1, type=int, help="The index of the text column in the file lists (default: 1)")
	parser.add_argument("--filelists", nargs="+", default=["data/msci-pt-dataset/msci-pt-raw-test-filelist.txt", "data/msci-pt-dataset/msci-pt-raw-train-filelist.txt", "data/msci-pt-dataset/msci-pt-raw-validation-filelist.txt"], help="The list of file lists to process")
	parser.add_argument("--text_cleaners", nargs="+", default=["msci_vits_pt"], help="The text cleaning methods to use")

	args = parser.parse_args()

	vocabulary = ""

	for filelist in args.filelists:

		print("START:", filelist)

		filepaths_and_text = load_filepaths_and_text(filelist)

		for i in range(len(filepaths_and_text)):

			print(f"Processing Line {i}...")

			original_text = filepaths_and_text[i][args.text_index]

			cleaned_text = text._clean_text(original_text, args.text_cleaners)

			vocabulary = list(set(cleaned_text) | set(vocabulary))

			filepaths_and_text[i][args.text_index] = cleaned_text

		new_filelist = filelist + "." + args.out_extension

		with open(new_filelist, "w", encoding="utf-8") as f:

			f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

	new_symbols = []

	for symbol in vocabulary:

		if symbol not in text.symbols:

			new_symbols.append(symbol)

	if len(new_symbols) > 0:

		print(f"The following symbols in the dataset are not in the vocabulary: {new_symbols}")

	else:

		print("All symbols in the dataset are in the vocabulary!")
