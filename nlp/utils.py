# coding: utf-8

"""
    description: NLP utils
    author: Suraj Iyer
"""

import re
from .trie import *
from collections import Counter
from os.path import dirname, join
import numpy as np
import distance
from sklearn.cluster import AffinityPropagation


def words(text): return re.findall(r'\w+', text.lower())

def build_words_from_corpus_file(file_path): return Counter(words(open(file_path).read()))

def build_words_from_dict_file(file_path, delimiter=' '):
	with open(file_path, 'r', encoding='utf8') as f:
		words = f.readlines()
	return set(line.split(delimiter)[0] for line in words)

def build_trie_from_dict_file(file_path, header=False, columns=[], delimiter=" ", cb=None):
	model = Trie()
	value = None

	with open(file_path, 'r', encoding='utf8') as f:

		# Handling headers in input file
		if header == 'include':
			columns = f.readline().replace('\n', '').split(delimiter)
		elif header == 'ignore':
			f.readline()
		start_pos = f.tell()

		# add all words to a Trie data structure
		if columns:
			model.addAll(((lambda x: {c: x[i] for i, c in enumerate(columns)})(line.replace('\n', '').split(delimiter)) for line in f.readlines()))
		else:
			model.addAll((line.replace('\n', '').split(delimiter)[0].lower() for line in f.readlines()))

		# call the callback function
		if cb:
			f.seek(start_pos)
			value = cb(f)

	if not cb:
		return model
	else:
		return model, value

def build_trie_from_corpus_file(file_path):
	lang_dict = Counter(words(open(file_path).read()))
	model = Trie()
	model.addAll(({'word': line[i], 'count': count} for word, count in Counter.items()))
	return model

def words_set_dictionary(lang='en', size='50k'):
	"""
	    lang: str, default='en'
	    	Currently only English (en) / Dutch (nl) supported.
	    size: str, default='50k'
	        Use the small dictionary containing only top '50k' words or 'full' dictionary.
	        Full dictionary is very large and can result in large running times.
    """
	return build_words_from_dict_file(join(dirname(__file__), lang, '{}_{}.txt'.format(lang, size)))

def words_trie_dictionary(lang='en', size='50k'):
	"""
	    lang: str, default='en'
	    	Currently only English (en) / Dutch (nl) supported.
	    size: str, default='50k'
	        Use the small dictionary containing only top '50k' words or 'full' dictionary.
	        Full dictionary is very large and can result in large running times.
    """
	return build_trie_from_dict_file(join(dirname(__file__), lang, '{}_{}.txt'.format(lang, size)), header='include')

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def reduce_lengthening(text):
	"""Correcting more than twice repeated characters in words."""
	pattern = re.compile(r"(.)\1{2,}")
	return pattern.sub(r"\1\1", text)

def edits_1(word):
    """All edits that are one edit away from `word`."""
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits_2(word):
    """All edits that are two edits away from `word`."""
    return (e2 for e1 in edits_1(word) for e2 in edits_1(e1))

def _edit_dist(word, dist=2, k=None):
    """All edits that are `n` edits away from `word`."""
    if k is None:
        k = dist
    assert dist > 0 and k > 0
    if k == 1:
        return edits_1(word)
    new_words = (w2 for w1 in edits_1(word) for w2 in _edit_dist(w1, dist, k-1))
    return list(set(new_words)) if k == dist else new_words

def edit_dist(word, dist=2):
    return _edit_dist(word, dist)

def cluster_words_by_edit_distance1(words, dist=2):
	cluster = dict()
	for i, w in enumerate(words):
		x = [words[j] for j in range(i+1, len(words)) if distance.levenshtein(w,words[j]) == 1]
		if len(x) > 0:
			cluster[w] = x
	return cluster

def cluster_words_by_edit_distance2(words, verbose=True, **kwargs):
	# from the paper: L Frey, Brendan J., and Delbert Dueck. "Clustering by passing messages between data points." 
	# science 315.5814 (2007): 972-976..
	lev_similarity = -1*np.array([[distance.levenshtein(w1, w2) for w1 in words] for w2 in words])
	affprop = AffinityPropagation(affinity="precomputed", **kwargs)
	affprop.fit(lev_similarity)
	clusters = dict()
	for cluster_id in np.unique(affprop.labels_):
		exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
		clusters[exemplar] = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
		if verbose:
			print(" - *%s:* %s" % (exemplar, ", ".join(clusters[exemplar])))
	return clusters

def count_words(sentence):
	return len(sentence.split(' '))
