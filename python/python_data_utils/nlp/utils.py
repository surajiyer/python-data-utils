# coding: utf-8

"""
    description: NLP utils
    author: Suraj Iyer
"""

__all__ = [
    'words',
    'create_dictionary',
    'words_set_dictionary',
    'words_trie_dictionary',
    'edits_1',
    'edits_2',
    '_edit_dist',
    'edit_dist',
    'cluster_words_by_edit_distance1',
    'cluster_words_by_edit_distance2',
    'count_words',
    'join_bigrams_with_replacements',
    'correct_word_compounding',
    'tf_idf',
    'replace_contractions',
    'knn_name_matching',
    'bigram_context',
    'cluster_urls',
    'corpus_level_tfidf',
    'extract_phrases'
]

import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
import nltk
from .trie import *
from .contractions import *
from python_data_utils.utils import load_artifact
import difflib
from typing import Iterable, Tuple


def words(text: str) -> str:
    return re.findall(r'\w+', text.lower())


def create_dictionary(
        corpus: str = None, file_path: str = None) -> Counter:
    if file_path:
        corpus = open(file_path).read()
    if not corpus:
        raise ValueError(
            'String corpus or file_path (path/to/corpus) must be given.')
    return Counter(words(corpus))


def words_set_dictionary(dictionary_name: str) -> set:
    """
        dictionary_name: str
            Name of the dictionary file to load words from.
            Available word dictionaries:
                1. english_dictionary_small
                2. english_dictionary_big
                3. dutch_dictionary_small
                4. dutch_dictionary_big
    """
    assert dictionary_name in (
        'english_dictionary_small', 'english_dictionary_big',
        'dutch_dictionary_small', 'dutch_dictionary_big')
    return set(load_artifact(dictionary_name)['word'])


def words_trie_dictionary(dictionary_name: str) -> Trie:
    """
        dictionary_name: str
            Name of the dictionary file to load words from.
            Available word dictionaries:
                1. english_dictionary_small
                2. english_dictionary_big
                3. dutch_dictionary_small
                4. dutch_dictionary_big
    """
    assert dictionary_name in (
        'english_dictionary_small', 'english_dictionary_big',
        'dutch_dictionary_small', 'dutch_dictionary_big')
    model = Trie()
    model.addAll(load_artifact(dictionary_name)['word'])
    return model


def edits_1(word: str) -> set:
    """All edits that are one edit away from `word`."""
    assert isinstance(word, str), '{} must be a string'.format(word)
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits_2(word: str) -> tuple:
    """All edits that are two edits away from `word`."""
    assert isinstance(word, str), '{} must be a string'.format(word)
    return tuple(e2 for e1 in edits_1(word) for e2 in edits_1(e1))


def _edit_dist(word: str, dist: int = 2, k: int = None) -> tuple:
    """All edits that are `n` edits away from `word`."""
    if k is None:
        k = dist
    assert dist > 0 and k > 0
    if k == 1:
        return edits_1(word)
    new_words = tuple(
        w2 for w1 in edits_1(word)
        for w2 in _edit_dist(w1, dist, k - 1))
    return tuple(set(new_words)) if k == dist else new_words


def edit_dist(word: str, dist: int = 2) -> tuple:
    assert isinstance(word, str), '{} must be a string'.format(word)
    assert isinstance(dist, int)
    return _edit_dist(word, dist)


def cluster_words_by_edit_distance1(
        words: Iterable[str], dist: int = 2) -> dict:
    assert isinstance(words, (list, tuple))
    assert any(isinstance(w, str) for w in words), 'words must be an iterable of strings'
    assert isinstance(dist, int)
    cluster = dict()
    import distance
    for i, w in enumerate(words):
        x = [words[j] for j in range(
            i + 1, len(words)) if distance.levenshtein(w, words[j]) == dist]
        if len(x) > 0:
            cluster[w] = x
    return cluster


def cluster_words_by_edit_distance2(
        words: Iterable[str], verbose: bool = True, **kwargs) -> dict:
    """
    Cluster words with affinity propagation using negative
    edit distance as similarity measure.
    """
    assert isinstance(words, (list, tuple)) and any(isinstance(w, str)
        for w in words), 'words must be an iterable of strings'
    assert isinstance(verbose, bool)

    # Compute edit distance between words
    import distance
    lev_similarity = -1 * \
        np.array([[distance.levenshtein(w1, w2)
                   for w1 in words] for w2 in words])

    # Compute word clusters with affinity propagation
    # using edit distance similarity between words as input.
    from python_data_utils.cluster.affinity_propagation import ap_precomputed
    return ap_precomputed(words, lev_similarity, verbose, **kwargs)


def count_words(sentence: str, delimiter: str = ' ') -> int:
    assert isinstance(sentence, str), '{} must be a string'.format(sentence)
    return len(sentence.split(delimiter))


def join_bigrams_with_replacements(
        s: Iterable[Iterable['str']],
        replacements=lambda w, default: default) -> str:
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
            Function that takes the following parameters. By default,
            it returns the default value which simply joins all the bigrams
            together.
                w: string
                    A bigram string. Example: "wi fi"
                default:
                    A default value to return in case no replacement available.
                returns: string
                    A replacement string, e.g., "wifi"
        returns: string
            A string sentence joining all bigrams correctly including the
            replacements (if any).
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


def correct_word_compounding(
        dfs: pd.Series, language_dictionary,
        split_on_both_accurate: bool = True,
        threshold: int = 3) -> pd.DataFrame:
    """
    List all bigrams which are also written as unigrams
    by removing one space character
    and filter the ones which contains words not in any dictionary

    Assumptions: every word is unique is only written in
    one way and there are no spelling errors

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
                        convert bigram to unigram by removing space
                        (make unibigram)
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
                    # Assumption: Unibigram version is not accurate so
                    # split instead
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

    bigrams = pd.DataFrame([(
        b,
        b.replace(" ", ""),
        b.replace(" ", "-"),
        dfs.str.contains(b).sum(),
        dfs.str.contains(b.replace(" ", "") + "|" + b.replace(" ", "-")).sum()
    ) for b in bigrams_also_in_unigrams], columns=[
        'bigram', 'unibigram0', 'unibigram1',
        'count_occurence_bigram', 'count_occurence_unibigram'])
    bigrams['near_equal'] = abs(
        bigrams["count_occurence_bigram"] -
        bigrams["count_occurence_unibigram"]) < threshold
    bigrams['unibigram_in_dict?'] = bigrams['bigram'].apply(
        lambda bi: bi.replace(" ", "") in language_dictionary)
    bigrams['unigram0_in_dict?'] = bigrams['bigram'].apply(
        lambda bi: bi.split(' ')[0] in language_dictionary)
    bigrams['unigram1_in_dict?'] = bigrams['bigram'].apply(
        lambda bi: bi.split(' ')[1] in language_dictionary)
    bigrams['both_unigrams_in_dict?'] = (
        bigrams['unigram0_in_dict?']) & (bigrams['unigram1_in_dict?'])
    make_bigram_mask = (
        (
            (bigrams["near_equal"]) &
            (
                (~bigrams["unibigram_in_dict?"]) |
                (bigrams["unibigram_in_dict?"] &
                    bigrams["both_unigrams_in_dict?"] &
                    split_on_both_accurate)
            )
        ) |
        (
            (~bigrams["near_equal"]) &
            (
                (
                    (bigrams["count_occurence_unibigram"] >
                        bigrams["count_occurence_bigram"]) &
                    (~bigrams["unibigram_in_dict?"]) &
                    (bigrams['both_unigrams_in_dict?'])
                ) |
                (
                    (bigrams["count_occurence_unibigram"] <=
                        bigrams["count_occurence_bigram"]) &
                    (bigrams['both_unigrams_in_dict?'])
                )
            )
        )
    )

    # List of word replacements to perform on the text
    replacements = {}
    replacements.update({
        w[1][f"unibigram{i}"]: w[1]["bigram"]
        for w in bigrams[make_bigram_mask].iterrows() for i in range(2)})
    replacements.update({
        w[1]["bigram"]: w[1]["unibigram0"]
        for w in bigrams[~make_bigram_mask].iterrows()})

    # Make replacements
    tokens = tokens.apply(lambda s: [replacements.get(w, w) for w in s])
    tokens.loc[tokens.apply(len) > 1] = tokens.loc[tokens.apply(len) > 1]\
        .apply(nltk.bigrams)\
        .apply(join_bigrams_with_replacements, replacements=replacements.get)
    tokens.loc[tokens.apply(len) == 1] = tokens.loc[tokens.apply(len) == 1]\
        .apply(lambda s: "".join(s))
    return tokens


def tf_idf(documents: Iterable[str]) -> Tuple[dict, dict, dict]:
    N = len(documents)
    assert N > 0, "Count of documents must be at least 1."
    corpus = " ".join(documents)
    corpus_deduplicated = " ".join(
        [" ".join(frozenset(d.split(" "))) for d in documents])

    tf = create_dictionary(corpus=corpus)
    df = create_dictionary(corpus=corpus_deduplicated)
    tf_idf = {t: tf * np.log(1 + N / df[t]) for t, tf in tf.items()}

    return tf_idf, tf, df


def replace_contractions(
        text: str,
        contractions: dict = english_contractions) -> str:
    """
    Based on https://gist.github.com/nealrs/96342d8231b75cf4bb82

    :param:
        text: str
            Input text
        contractions: dict (default=english_contractions)
            Dictionary of contractions (key) and their expanded forms (value).
    """
    c_re = re.compile(r'\b(%s)\b' % '|'.join(contractions.keys()))

    def replace(match):
        return contractions[match.group(0)]

    return c_re.sub(replace, text)


def knn_name_matching(
        A: Iterable[str], B: Iterable[str],
        vectorizer_kws: dict = {}, nn_kws: dict = {},
        max_distance: float = None, return_B=True) -> list:
    """
    Nearest neighbor name matching of sentences in B to A.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.feature_extraction.text import TfidfVectorizer

    # vectorize the B documents after fitting on A
    vectorizer = TfidfVectorizer(**vectorizer_kws)
    Xa = vectorizer.fit_transform(A)
    Xb = vectorizer.transform(B)

    # find nearest neighbor matching
    neigh = NearestNeighbors(n_neighbors=1, **nn_kws)
    neigh.fit(Xa)
    if max_distance is None:
        indices = neigh.kneighbors(Xb, return_distance=False).flatten()
    else:
        indices, distances = neigh.kneighbors(Xb)
        indices, distances = indices.flatten(), distances.flatten()
        indices = indices[distances <= max_distance]

    if return_B:
        result = [(B[i], A[idx]) for i, idx in enumerate(indices)]
    else:
        result = [A[idx] for idx in indices]

    return result


def bigram_context(
        documents: Iterable[str], window_size: int = 2,
        unique: bool = False, sort: bool = True) -> pd.DataFrame:
    """
    Get bigram context (word pairs) within given window size
    from given documents
    """
    words = []
    for d in documents:
        words.extend(d)
        words.extend(["|"] * (window_size - 1))
    finder = nltk.collocations.BigramCollocationFinder\
        .from_words(words, window_size=window_size)
    finder.apply_ngram_filter(lambda *w: "|" in w)
    df = pd.DataFrame(finder.ngram_fd.items(), columns=["pair", "count"])
    df[["left", "right"]] = df['pair'].apply(pd.Series)
    df.drop("pair", axis=1, inplace=True)
    if unique:
        df.loc[df.left >= df.right, ['left', 'right']] =\
            df.loc[df.left >= df.right, ['right', 'left']].values
        df = df.groupby(['left', 'right'])['count'].sum().reset_index()
    if sort:
        df.sort_values("count", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def cluster_urls(
        urls: Iterable[str],
        min_cluster_size: int = 10) -> pd.DataFrame:
    """
    Cluster URLs by regex rules defined in this package:
    https://pypi.org/project/urlclustering/

    urls: list
        List of urls.
    min_clustre_size: int
        Minimum cluster size
    """
    import urlclustering
    clusters = urlclustering.cluster(urls, min_cluster_size)
    tmp = {v0: [k[1], 0] for k, v in clusters['clusters'].items() for v0 in v}
    tmp.update({k: [k, 1] for k in clusters['unclustered']})
    clusters = pd.DataFrame.from_dict(
        tmp, orient='index', columns=['cluster', 'unclustered'])
    return clusters


def corpus_level_tfidf(
        texts: Iterable[str], **vectorizer_kwargs) -> Tuple[Iterable[float], Iterable[str]]:
    """
    See book: Ziegler, C.N., 2012. Mining for strategic competitive
    intelligence. Springer. pp. 128-130.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    with np.errstate(divide='ignore'):
        # Compute 1 + log(tf+(t_i)) where tf+ computes the summed term
        # frequency for term t_i and t_i occurs in some text `d` \in texts.
        vectorizer_kwargs.pop('min_df', None)
        vectorizer_kwargs.pop('max_df', None)
        tf_vect = CountVectorizer(**vectorizer_kwargs)
        tf = np.array(tf_vect.fit_transform(texts).sum(0)).reshape(-1)
        tf = (tf > 0).astype(float) + np.log(tf, where=tf > 0)

        # Get the order of the terms in the tf_vect
        terms = tf_vect.get_feature_names()

        # Compute log(|texts| / df(t_i)) where df computes the document
        # frequency of t_i in texts.
        vectorizer_kwargs.pop('vocabulary', None)
        vectorizer_kwargs.pop('binary', None)
        idf_vect = CountVectorizer(
            binary=True, vocabulary=terms, **vectorizer_kwargs)
        idf = np.array(np.log(
            len(texts) / (1. + idf_vect.fit_transform(texts).sum(0))
        )).reshape(-1)
        return tf * idf, terms


def extract_phrases(
        list_of_strings: Iterable[str], min_phrase_length: int = 2,
        max_phrase_length: int = float("inf"), min_freq: int = 1,
        delimiter: str = "||") -> dict:
    N = len(list_of_strings)
    assert N > 0
    phrases = defaultdict(int)
    string = list_of_strings[0]
    if N == 1:
        phrases[string] = 1
    else:
        for i in range(1, len(list_of_strings)):
            matches = difflib\
                .SequenceMatcher(None, list_of_strings[i], string)\
                .get_matching_blocks()[:-1]
            string += delimiter
            end = 0
            for match in matches:
                string += " " + list_of_strings[i][end:match.a]
                end = match.a + match.size
                phrases[list_of_strings[i][match.a:end]] += 1
            string += " " + list_of_strings[i][end:]
    phrases = {
        k: v for k, v in map(lambda i: (i[0].strip(), i[1]), phrases.items())
        if (v >= min_freq) and
        (min_phrase_length <= len(k) <= max_phrase_length)}
    return phrases
