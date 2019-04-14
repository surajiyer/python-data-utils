# coding: utf-8

"""
    description: NLP utils
    author: Suraj Iyer
"""

import re
from collections import Counter


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def reduce_lengthening(text):
	"""Correcting more than twice repeated characters in words."""
	pattern = re.compile(r"(.)\1{2,}")
	return pattern.sub(r"\1\1", text)

def words(text): return re.findall(r'\w+', text.lower())

def read_count_words(filepath): return Counter(words(open(filepath).read()))