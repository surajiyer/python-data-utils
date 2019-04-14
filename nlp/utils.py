# coding: utf-8

"""
    description: NLP utils
    author: Suraj Iyer
"""

import re
from collections import Counter
import numpy as np
import distance
from sklearn.cluster import AffinityPropagation


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

def cluster_words_by_edit_distance1(words, edit_distance=2):
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
