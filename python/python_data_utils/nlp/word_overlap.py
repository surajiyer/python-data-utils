# coding: utf-8

"""
    description: String overlap functions
    author: Suraj Iyer
"""

__all__ = ['WordOverlap']

import nltk
import numpy as np


class WordOverlap:

    @staticmethod
    def BLEU_score(text_1: str, text_2: str, n: int = 1):
        len_1 = len(text_1)
        len_2 = len(text_2)
        print(len_1, len_2)
        sum_p = 0
        for i in range(1, n + 1):
            print(i)
            n_grams_1 = set(nltk.ngrams(text_1.split(), i))
            n_grams_2 = set(nltk.ngrams(text_2.split(), i))
            print(n_grams_1, n_grams_2)
            p = len(n_grams_1.intersection(n_grams_2))
            print(p)
            sum_p += np.log(p)
        print(sum_p)
        print(np.exp(sum_p / n))
        np.exp(1 - max(1, len_1 / len_2))
        return np.exp(1 - max(1, len_1 / len_2)) * np.exp(sum_p / n)
