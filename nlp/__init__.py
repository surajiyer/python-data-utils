# coding: utf-8

"""
    description: NLP package
    author: Suraj Iyer
"""

from os.path import dirname, join
from . import utils
from .spell import SpellCheck


def nl_50k():
	# Load the nl_50k dutch dataset
	with open(join(dirname(__file__), 'nl', 'nl_50k.txt'), 'r') as f:
	    nl_50k = f.readlines()
	return [line.split(' ')[0] for line in nl_50k]

def nl_full():
	# Load the nl_50k dutch dataset
	with open(join(dirname(__file__), 'nl', 'nl_full.txt'), 'r') as f:
	    nl_50k = f.readlines()
	return [line.split(' ')[0] for line in nl_50k]
