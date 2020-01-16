# coding: utf-8

"""
    description: keyphrase extraction functions
    author: Suraj Iyer
"""

__all__ = ['KeyphraseExtraction']

import utils as nlpu
import numpy as np
from typing import Iterable


class KeyphraseExtraction:

    @staticmethod
    def ziegler_foreground_keywords_extraction(
            D_f: Iterable[str], D_b: Iterable[str],
            **vectorizer_kwargs) -> Iterable[str]:
        """
        Keyphrase extraction by contrasting foreground and background
        aggregated tf-idf weights of terms to enable context awareness.

        See paper: Ziegler, C.N., Skubacz, M. and Viermetz, M., 2008, December.
        Mining and exploring unstructured customer feedback data using language
        models and treemap visualizations. In 2008 IEEE/WIC/ACM International
        Conference on Web Intelligence and Intelligent Agent Technology (Vol. 1,
        pp. 932-937). http://www2.informatik.uni-freiburg.de/~cziegler/papers/WI-08-CR.pdf
        """
        # TODO: the extracted keywords are NOT good quality, needs more improvement
        weights_f, terms_f = nlpu.corpus_level_tfidf(D_f, **vectorizer_kwargs)
        vectorizer_kwargs.update({'vocabulary': terms_f})
        weights_b, _ = nlpu.corpus_level_tfidf(D_b, **vectorizer_kwargs)
        W = (weights_f / weights_b) * np.log(weights_f + weights_b)

        # if a term occurs in the foreground corpus but not in the background,
        # then the term weight becomes +infinity. This does not give useful
        # information about how to order the terms based on their importance,
        # so we fall back to using its foreground weights only in this case.
        mask = np.isposinf(W)
        W[mask] = weights_f[mask]

        # print('\n'.join([str(x) for x in list(zip(terms_f, W, weights_f, weights_b))]))
        return np.array(terms_f)[np.argsort(-W)]
