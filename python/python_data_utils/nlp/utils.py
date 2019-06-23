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
import nltk
import pandas as pd
from .contractions import *


def words(text):
    return re.findall(r'\w+', text.lower())


def create_dictionary(corpus=None, file_path=None):
    if file_path:
        corpus = open(file_path).read()
    if not corpus:
        raise ValueError(
            'String corpus or file_path (path/to/corpus) must be given.')
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
            Use the small dictionary containing only top '50k'
            words or 'full' dictionary. Full dictionary is very large
            and can result in large running times.
    """
    return join(words_data_folder_path(lang), '{}_{}.txt'.format(lang, size))


def words_dictionary_trie_filepath(lang='en', size='50k'):
    """
        lang: str, default='en'
            Currently only English (en) / Dutch (nl) supported.
        size: str, default='50k'
            Use the small dictionary containing only top '50k'
            words or 'full' dictionary. Full dictionary is very large
            and can result in large running times.
    """
    return join(words_data_folder_path(lang), '{}_{}_trie'.format(lang, size))


def words_set_dictionary(lang='en', size='50k'):
    """
        lang: str, default='en'
            Currently only English (en) / Dutch (nl) supported.
        size: str, default='50k'
            Use the small dictionary containing only top '50k'
            words or 'full' dictionary. Full dictionary is very large
            and can result in large running times.
    """
    return create_dictionary_from_csv(words_dictionary_filepath(lang, size))


def words_trie_dictionary(lang='en', size='50k'):
    """
        lang: str, default='en'
            Currently only English (en) / Dutch (nl) supported.
        size: str, default='50k'
            Use the small dictionary containing only top '50k'
            words or 'full' dictionary. Full dictionary is very large
            and can result in large running times.
    """
    return create_trie_dictionary_from_csv(
        words_dictionary_filepath(lang, size), header='include')


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


def cluster_words_by_edit_distance2(words, verbose=True, **kwargs):
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
    from ..cluster import affinity_propagation
    return affinity_propagation(words, lev_similarity, verbose, **kwargs)


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


def correct_word_compounding(dfs, language_dictionary,
                             split_on_both_accurate=True, threshold=3):
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
        w[1]["unibigram{}".format(i)]: w[1]["bigram"]
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


def tf_idf(documents):
    N = len(documents)
    corpus = " ".join(documents)
    corpus_deduplicated = " ".join(
        [" ".join(frozenset(d.split(" "))) for d in documents])

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
