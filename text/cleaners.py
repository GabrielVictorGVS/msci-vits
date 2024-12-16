import re
from unidecode import unidecode
from phonemizer import phonemize
from num2words import num2words

from phonemizer.backend.espeak.wrapper import EspeakWrapper

def expand_abbreviations(text):

	abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
		('dr', 'doutor'),
        ('dra', 'doutora'),
        ('sr', 'senhor'),
        ('sra', 'senhora'),
        ('prof', 'professor'),
        ('profa', 'professora'),
        ('msc', 'mestrado'),
        ('phd', 'doutorado'),
        ('jr', 'junior'),
        ('st', 'santo'),
        ('sta', 'santa'),
        ('av', 'avenida'),
        ('nº', 'número'),
        ('lt', 'lote'),
        ('sgt', 'sargento'),
        ('cap', 'capitão'),
        ('maj', 'major'),
        ('gen', 'general'),
        ('drs', 'doutores'),
        ('rev', 'reverendo'),
        ('exmo', 'excelentíssimo'),
        ('exma', 'excelentíssima'),
        ('ct', 'comandante'),
        ('eq', 'esquadrão'),
        ('eng', 'engenheiro'),
        ('me', 'mestre'),
        ('v. exa', 'vossa excelência'),
        ('etc', 'etcetera'),
	]]
  
	for regex, replacement in abbreviations:

		text = re.sub(regex, replacement, text)

	return text

def expand_numbers(text):

	def convert_numbers(match):

		numero = int(match.group())

		return num2words(numero, lang='pt') 

	text = re.sub(r'\d+', convert_numbers, text)

	return text

def msci_vits_pt(text: str):

    text = text.lower()

    text = expand_abbreviations(text)

    text = expand_numbers(text)

    phonemes = phonemize(text, language='pt-br', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)

    phonemes = re.sub(re.compile(r'\s+'), ' ', phonemes)

    return phonemes
