# coding: utf-8

"""
    description: NLP package
    author: Suraj Iyer
"""

from os.path import dirname, join
from . import utils
from .spell import SpellCheck


def word_corpus(lang='nl', size='50k'):
	with open(join(dirname(__file__), lang, '{}_{}.txt'.format(lang, size)), 'r', encoding='utf8') as f:
	    words = f.readlines()
	return [line.split(' ')[0] for line in words]
