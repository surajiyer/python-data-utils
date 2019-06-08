# coding: utf-8

"""
    description: NLP utils
    author: Suraj Iyer
"""

import re
from python_data_utils.nlp.trie import *
from collections import Counter
from os.path import dirname, join
import numpy as np
import nltk
import pandas as pd
from python_data_utils.nlp.contractions import *


def words(text): return re.findall(r'\w+', text.lower())


def create_dictionary(corpus=None, file_path=None):
    if file_path:
        corpus = open(file_path).read()
    if not corpus:
        raise ValueError('String corpus or file_path (path/to/corpus) must be given.')
    return Counter(words(corpus))


def create_dictionary_from_csv(file_path, header=False, delimiter=" "):
    with open(file_path, 'r', encoding='utf8') as f:
        if header:
            f.readline()
        words = f.readlines()
    return set(line.split(delimiter)[0] for line in words)


def create_trie_dictionary(corpus=None, file_path=None):
    if filepath:
        corpus = open(file_path).read()
    lang_dict = create_dictionary(corpus)
    model = Trie()
    model.addAll(({'word': word, 'count': count}
                  for word, count in lang_dict.items()))
    return model


def words_data_folder_path(lang='en'):
    return join(dirname(__file__), 'data', 'dictionaries', lang)


def words_dictionary_filepath(lang='en', size='50k'):
    """
        lang: str, default='en'
            Currently only English (en) / Dutch (nl) supported.
        size: str, default='50k'
            Use the small dictionary containing only top '50k' words or 'full' dictionary.
            Full dictionary is very large and can result in large running times.
    """
    return join(words_data_folder_path(lang), '{}_{}.txt'.format(lang, size))


def words_dictionary_trie_filepath(lang='en', size='50k'):
    """
        lang: str, default='en'
            Currently only English (en) / Dutch (nl) supported.
        size: str, default='50k'
            Use the small dictionary containing only top '50k' words or 'full' dictionary.
            Full dictionary is very large and can result in large running times.
    """
    return join(words_data_folder_path(lang), '{}_{}_trie'.format(lang, size))


def words_set_dictionary(lang='en', size='50k'):
    """
        lang: str, default='en'
            Currently only English (en) / Dutch (nl) supported.
        size: str, default='50k'
            Use the small dictionary containing only top '50k' words or 'full' dictionary.
            Full dictionary is very large and can result in large running times.
    """
    return create_dictionary_from_csv(words_dictionary_filepath(lang, size))


def words_trie_dictionary(lang='en', size='50k'):
    """
        lang: str, default='en'
            Currently only English (en) / Dutch (nl) supported.
        size: str, default='50k'
            Use the small dictionary containing only top '50k' words or 'full' dictionary.
            Full dictionary is very large and can result in large running times.
    """
    return create_trie_dictionary_from_csv(words_dictionary_filepath(lang, size), header='include')


def cleanhtml(raw_html):
    assert isinstance(raw_html, str), '{} must be a string'.format(raw_html)
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def reduce_lengthening(text):
    """Correcting more than twice repeated characters in words."""
    assert isinstance(text, str), '{} must be a string'.format(text)
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def edits_1(word):
    """All edits that are one edit away from `word`."""
    assert isinstance(word, str), '{} must be a string'.format(word)
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits_2(word):
    """All edits that are two edits away from `word`."""
    assert isinstance(word, str), '{} must be a string'.format(word)
    return (e2 for e1 in edits_1(word) for e2 in edits_1(e1))


def _edit_dist(word, dist=2, k=None):
    """All edits that are `n` edits away from `word`."""
    if k is None:
        k = dist
    assert dist > 0 and k > 0
    if k == 1:
        return edits_1(word)
    new_words = (w2 for w1 in edits_1(word)
                 for w2 in _edit_dist(w1, dist, k - 1))
    return list(set(new_words)) if k == dist else new_words


def edit_dist(word, dist=2):
    assert isinstance(word, str), '{} must be a string'.format(word)
    assert isinstance(dist, int)
    return _edit_dist(word, dist)


def cluster_words_by_edit_distance1(words, dist=2):
    assert isinstance(words, (list, tuple)) and any(isinstance(w, str)
                                                    for w in words), 'words must be an iterable of strings'
    assert isinstance(dist, int)
    cluster = dict()
    for i, w in enumerate(words):
        x = [words[j] for j in range(
            i + 1, len(words)) if distance.levenshtein(w, words[j]) == dist]
        if len(x) > 0:
            cluster[w] = x
    return cluster


def documents_clustering_affinity_propagation(documents, similarity_matrix, document_id_included=False, verbose=True, **kwargs):
    """
    Create clusters with affinity propagation using given similarity matrix between documents as input.
    """
    from sklearn.cluster import AffinityPropagation
    documents = np.array(documents)
    affprop = AffinityPropagation(affinity="precomputed", **kwargs)
    affprop.fit(similarity_matrix)
    clusters = dict()
    for cluster_id in np.unique(affprop.labels_):
        exemplar = documents[affprop.cluster_centers_indices_[cluster_id]]
        if document_id_included:
            exemplar = exemplar[0]
            clusters[exemplar] = frozenset([idx for idx, _ in documents[np.flatnonzero(affprop.labels_ == cluster_id)]])
        else:
            clusters[exemplar] = frozenset([d for d in documents[np.flatnonzero(affprop.labels_ == cluster_id)]])
        if verbose:
            print(" - *%s:* %s" % (exemplar, ", ".join(str(d) for d in clusters[exemplar])))
    return clusters


def cluster_words_by_edit_distance2(words, verbose=True, **kwargs):
    """
    Paper:
        L Frey, Brendan J., and Delbert Dueck. "Clustering by passing messages between data points."
        science 315.5814 (2007): 972-976..
    """
    assert isinstance(words, (list, tuple)) and any(isinstance(w, str)
                                                    for w in words), 'words must be an iterable of strings'
    assert isinstance(verbose, bool)

    # Compute edit distance between words
    import distance
    lev_similarity = -1 * np.array([[distance.levenshtein(w1, w2) for w1 in words] for w2 in words])

    # Compute word clusters with affinity propagation using edit distance similarity between words as input.
    return documents_clustering_affinity_propagation(words, lev_similarity, verbose, **kwargs)


def documents_similarity_jaccard_affinity(documents, verbose=True, **kwargs):
    """
    Cluster text documents with affinity propagation based on jaccard similarity scores.

    :param documents: list of tuples of type [(int, str),...]
        Each tuple in list of documents is a pair of document id (int) and document text (str).
    """
    assert isinstance(verbose, bool)

    # Computer jaccard similarity between documents
    import distance
    from python_data_utils.numpy_utils import create_symmetric_matrix
    documents = np.array([(idx, set(doc.split(" "))) for idx, doc in documents])
    jaccard_similarity = [0 if idx1 == idx2 else -1 * distance.jaccard(doc1, doc2) for idx1, doc1 in documents for idx2, doc2 in documents if idx1 <= idx2]
    jaccard_similarity = create_symmetric_matrix(jaccard_similarity)

    # Create clusters with affinity propagation using jaccard similarity between documents as input.
    return documents_clustering_affinity_propagation(documents, jaccard_similarity, True, verbose, **kwargs)


def unilateral_jaccard(documents: list, depth: int=1) -> np.ndarray:
    """
    More information in the paper: Unilateral Jaccard Similarity Coefficient
    https://pdfs.semanticscholar.org/3031/c9f0c265571846cc2bfd7d8ca3538918a355.pdf

    Compute a graph G where nodes = documents and edges represents intersection in words occurring between documents.
    The uJaccard similarity score is the number of paths in G between pairs of documents within maximum :depth: distance.

    uJaccard(A, B) = |paths(A, B, depth)| / |edges(A)|

    :param documents:  list of sets [set,...]
        The sets represent a single document. The elements of the set are essentially the "words".
    :param depth: int
        Maximum path length between documents
    """
    document_edges = [int(len(doc1.intersection(doc2)) > 0) if i != i + j else 0 for i, doc1 in enumerate(documents) for j, doc2 in enumerate(documents[i:])]
    from python_data_utils.numpy_utils import create_symmetric_matrix
    document_edges = create_symmetric_matrix(document_edges)
    uJaccard = np.full((len(documents),) * 2, -1.)

    def get_all_paths_util(u, v, edges, depth, all_paths, visited, path):
        # If exceeds depth, return without doing anything
        if depth == 0:
            return

        # Mark the current node as visited and store in path
        visited[u] = True
        path.append(u)

        # If current vertex is same as destination, then print
        # append the path to all paths list
        if u == v:
            all_paths.append(path[:])
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i, e in enumerate(edges[u]):
                if e == 1 and not visited[i]:
                    get_all_paths_util(i, v, edges, depth - 1, all_paths, visited, path)

        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u] = False

    def n_paths_u_to_v(u, v, edges, depth):
        assert depth > 0
        visited = [False] * len(edges[0])
        paths_to_v = []
        get_all_paths_util(u, v, edges, depth, paths_to_v, visited, [])
        return len(paths_to_v)

    for i, j in np.ndindex(*uJaccard.shape):
        # Similarity score between same nodes is always 100%
        if i == j:
            uJaccard[i, j] = 1.
            continue

        # Compute the number of outgoing edges from node i
        n_edges = sum(document_edges[i])

        # Check > 0 to avoid div by 0.
        if n_edges > 0:
            # # paths from i to j equals # paths from j to i, therefore, compute # paths
            # from i to j if # paths from j to i is not already computed else use that
            x = n_paths_u_to_v(i, j, document_edges, depth) if uJaccard[j, i] == -1. else uJaccard[j, i]
            uJaccard[i, j] = x / n_edges
        else:
            uJaccard[i, j] = 0.

    return uJaccard


def documents_similarity_ujaccard_affinity(documents, depth=3, verbose=True, **kwargs):
    """
    Cluster text documents with affinity propagation based on Unilateral Jaccard similarity scores.

    :param documents: list of tuples of type [(int, str),...]
        Each tuple in list of documents is a pair of document id (int) and document text (str).
    """
    assert isinstance(verbose, bool)

    # Computer jaccard similarity between documents
    uJaccard_similarity = unilateral_jaccard([set(doc.split(" ")) for _, doc in documents], depth=depth)
    return documents_clustering_affinity_propagation(documents, uJaccard_similarity, True, verbose, **kwargs)


def count_words(sentence, delimiter=' '):
    assert isinstance(sentence, str), '{} must be a string'.format(sentence)
    return len(sentence.split(delimiter))


def join_bigrams_with_replacements(s, replacements=lambda w, default: default):
    """
    Function to join a list of bigrams back into a sentence taking into account
    unigram replacements for bigrams.

    working version: 1.0
    # does not work with generator input
    skip = False
    string = ""
    n = len(s)
    for i, w in enumerate(s):
        if skip:
            skip = False
            if i == n-1:
                string = string + " " + w[1]
            continue
        w = replacements(" ".join(w), w)
        if isinstance(w, str):
            string = string + " " + w
            skip = True
        else:
            string = string + " " + w[0]
    return string

    param:
        s: iterable of list of strings
            Example: [("I", "have"), ("have", "to"), ("to", "go")]
        replacements:
            Function that takes the following parameters. By default, it returns the default
            value which simply joins all the bigrams together.
                w: string
                    A bigram string. Example: "wi fi"
                default:
                    A default value to return in case no replacement available.
                returns: string
                    A replacement string, e.g., "wifi"
        returns: string
            A string sentence joining all bigrams correctly including the replacements (if any).
    """
    s = iter(s)
    first = next(s, None)
    if first is None:
        return ""
    first = replacements(" ".join(first), first)
    string = first if isinstance(first, str) else first[0] + " " + first[1]
    prev_string = "" if isinstance(first, str) else first[0]
    for w in s:
        w = replacements(" ".join(w), w)
        if isinstance(w, str):
            string = prev_string + " " + w
            prev_string = string
        else:
            prev_string = string
            string += " " + w[1]
    return string


def correct_word_compounding(dfs, language_dictionary, split_on_both_accurate=True, threshold=3):
    """
    List all bigrams which are also written as unigrams by removing one space character
    and filter the ones which contains words not in any dictionary

    Assumptions: every word is unique is only written in one way and there are no spelling errors

    Heuristic:
        if bigram counts ≈ unibigram counts:
            # Assumption: both versions are accurate enough
            if not unibigram in language dictionary:
                # Assumption: only bigram version accurate
                split unibigrams to unigrams (make bigram)
            else:
                # Assumption: both versions are still accurate enough
                if both unigrams in language dictionary:
                    # Assumption: both versions are still accurate enough
                    # so depend on {:split_on_both_accurate:}
                    if split_on_both_accurate:
                        make bigram
                    else:
                        convert bigram to unigram by removing space (make unibigram)
                else:
                    # Assumption: only unibigram version accurate
                    make unibigram
        else:
            # Assumption: Only one version is accurate
            if unibigram count >> bigram count:
                # Assumption: Unibigram version is more accurate
                if unibigram in language dictionary:
                    # Assumption: Unibigram version is still more accurate
                    make unibigram
                else:
                    # Assumption: Unibigram version is not accurate so split instead
                    if both unigrams in language dictionary:
                        make bigram
                    else:
                        make unibigram
            else:
                # Assumption: Bigram version is more accurate
                if both unigrams in language dictionary:
                        make bigram
                    else:
                        make unibigram
    """
    assert isinstance(threshold, int) and threshold > 0
    assert isinstance(split_on_both_accurate, bool)
    tokens = dfs.str.split(' ')
    unigrams = set(w for s in tokens for w in s)
    bigrams = set(" ".join(w) for s in tokens.apply(nltk.bigrams) for w in s)
    bigrams_also_in_unigrams = set(
        b for b in bigrams if b.replace(" ", "") in unigrams)

    bigrams = pd.DataFrame([(b, b.replace(" ", ""), b.replace(" ", "-"), dfs.str.contains(b).sum(), dfs.str.contains(b.replace(" ", "") + "|" + b.replace(" ", "-")).sum())
                            for b in bigrams_also_in_unigrams], columns=['bigram', 'unibigram0', 'unibigram1', 'count_occurence_bigram', 'count_occurence_unibigram'])
    bigrams['near_equal'] = abs(
        bigrams["count_occurence_bigram"] - bigrams["count_occurence_unibigram"]) < threshold
    bigrams['unibigram_in_dict?'] = bigrams['bigram'].apply(
        lambda bi: bi.replace(" ", "") in language_dictionary)
    bigrams['unigram0_in_dict?'] = bigrams['bigram'].apply(
        lambda bi: bi.split(' ')[0] in language_dictionary)
    bigrams['unigram1_in_dict?'] = bigrams['bigram'].apply(
        lambda bi: bi.split(' ')[1] in language_dictionary)
    bigrams['both_unigrams_in_dict?'] = (
        bigrams['unigram0_in_dict?']) & (bigrams['unigram1_in_dict?'])
    make_bigram_mask = (((bigrams["near_equal"]) & ((~bigrams["unibigram_in_dict?"])
        | (bigrams["unibigram_in_dict?"] & bigrams["both_unigrams_in_dict?"] & split_on_both_accurate)))
    | ((~bigrams["near_equal"]) & (((bigrams["count_occurence_unibigram"] > bigrams["count_occurence_bigram"])
        & (~bigrams["unibigram_in_dict?"]) & (bigrams['both_unigrams_in_dict?']))
    | ((bigrams["count_occurence_unibigram"] <= bigrams["count_occurence_bigram"])
        & (bigrams['both_unigrams_in_dict?'])))))

    # List of word replacements to perform on the text
    replacements = {}
    replacements.update({w[1]["unibigram{}".format(i)]: w[1]["bigram"]
                         for w in bigrams[make_bigram_mask].iterrows() for i in range(2)})
    replacements.update({w[1]["bigram"]: w[1]["unibigram0"]
                         for w in bigrams[~make_bigram_mask].iterrows()})

    # Make replacements
    tokens = tokens.apply(lambda s: [replacements.get(w, w) for w in s])
    tokens.loc[tokens.apply(len) > 1] = tokens.loc[tokens.apply(len) > 1]\
        .apply(nltk.bigrams)\
        .apply(join_bigrams_with_replacements, replacements=replacements.get)
    tokens.loc[tokens.apply(len) == 1] = tokens.loc[tokens.apply(len) == 1]\
        .apply(lambda s: "".join(s))
    return tokens


def tf_idf(documents):
    N = len(documents)
    corpus = " ".join(documents)
    corpus_deduplicated = " ".join([" ".join(frozenset(d.split(" "))) for d in documents])

    tf = create_dictionary(corpus=corpus)
    df = create_dictionary(corpus=corpus_deduplicated)
    tf_idf = {t: tf * np.log(1 + N / df[t]) for t, tf in tf.items()}

    return tf_idf, tf, df


def replace_contractions(text, cDict=english_contractions):
    """
    Based on https://gist.github.com/nealrs/96342d8231b75cf4bb82

    :param:
        text: str
            Input text
        cDict: dict (default=english_contractions)
            Dictionary of contractions (key) and their expanded forms (value).
    """
    c_re = re.compile(r'\b(%s)\b' % '|'.join(cDict.keys()))

    def replace(match):
        return cDict[match.group(0)]

    return c_re.sub(replace, text)


class RegexPattern:
    WholeWordOnly = lambda w: r'\b{}\b'.format(w)
    Linebreak = r'\r+|\n+'
    Number = WholeWordOnly(r"[0-9]+([.,][0-9]+)?")
    TwoOrMoreSpaces = r'\s{2,}'
    Email = WholeWordOnly(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    NL_pc4 = WholeWordOnly(r'[0-9]{4}\s?\w{2}')
    SingleCharacterWord = WholeWordOnly(r'\w')
    TimeOfDay = r"([0[0-9]|1[0-9]|2[0-3]):[0-5][0-9](\s?a[.]?m[.]?|p[.]?m[.]?|A[.]?M[.]?|P[.]?M[.]?)?"
    Unicode = r"[^\x00-\x7F]"
    Quotes = lambda x: r"(\"{0}\"|\'{0}\')".format(x)
    URL = r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?'
    SpecialCharacters = r'[`!@#*-+{}\[\]:;\'"|\\,<>\/]'
    NumbersWithSuffix = r'[0-9]+(st|th|nd|rd)'
    VersionNumber3N = r"\b(\d+\.)?(\d+\.)?(\*|\d+)\b"
    Copyright = r"\(c\)|®|©|™"
    ThreePlusRepeatingCharacters = r"([a-z])\1{2,}"
    ApostropheWords = r"[\w]+['][\w]+(['][\w]+)?"
