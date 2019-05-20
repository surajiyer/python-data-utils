# coding: utf-8

"""
    description:
        Scikit-learn compatible implementation of the Gibberish detector
        based on https://github.com/rrenaud/Gibberish-Detector
    original author: rrenaud@github
    author: Suraj Iyer
"""

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
import numpy as np


class GibberishDetectorClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, accepted_chars='abcdefghijklmnopqrstuvwxyz ', smoothing_factor=10):
        self.accepted_chars = accepted_chars
        self.smoothing_factor = smoothing_factor

    def set_params(self, **params):
        return super(GibberishDetectorClassifier, self).set_params(**params)

    @property
    def accepted_chars(self):
        return self._accepted_chars

    @accepted_chars.setter
    def accepted_chars(self, value):
        self._accepted_chars = value
        self._pos = dict([(char, idx) for idx, char in enumerate(value)])

    def _normalize(self, line):
        """ Return only the subset of chars from accepted_chars.
        This helps keep the  model relatively small by ignoring punctuation, infrequent symbols, etc. """
        return [c.lower() for c in line if c.lower() in self.accepted_chars]

    def _ngram(self, n, line):
        """ Return all n grams from line after normalizing """
        filtered = self._normalize(line)
        for start in range(0, len(filtered) - n + 1):
            yield ''.join(filtered[start:start + n])

    def fit(self, X, y=None):
        """ Write a simple model as a pickle file """
        k = len(self._accepted_chars)

        # Assume we have seen `self.smoothing_factor` of each character pair.
        # This acts as a kind of prior or smoothing factor. This way, if we see a
        # character transition live that we've never observed in the past, we won't
        # assume the entire string has 0 probability.
        counts = [[self.smoothing_factor for i in range(k)] for i in range(k)]

        # Count transitions between characters in lines from X, taken
        # from http://norvig.com/spell-correct.html
        for line in X:
            for a, b in self._ngram(2, line):
                counts[self._pos[a]][self._pos[b]] += 1

        # _normalize the counts so that they become log probabilities.
        # We use log probabilities rather than straight probabilities to avoid
        # numeric underflow issues with long texts.
        # This contains a justification:
        # http://squarecog.wordpress.com/2009/01/10/dealing-with-underflow-in-joint-probability-calculations/
        for i, row in enumerate(counts):
            s = float(sum(row))
            for j in range(len(row)):
                row[j] = np.log(row[j] / s)

        self._log_prob_mat = counts

        return self

    def _avg_transition_prob(self, line):
        """ Return the average transition probability of line with the log probability matrix. """
        log_prob = 0.0
        transition_ct = 0

        for a, b in self._ngram(2, line):
            log_prob += self._log_prob_mat[self._pos[a]][self._pos[b]]
            transition_ct += 1

        # The exponentiation translates from log probability to regular probability.
        return np.exp(log_prob / (transition_ct or 1))

    def predict_proba(self, X):
        check_is_fitted(self, '_log_prob_mat')
        return np.array([self._avg_transition_prob(x) for x in X])

    def predict(self, X, threshold):
        # if the transition probability is lower than threshold, its gibberish, i.e., return 1 else 0
        return (self.predict_proba(X) < threshold) * 1


if __name__ == '__main__':
    import utils

    # load words
    words = utils.create_dictionary_from_csv(utils.words_dictionary_filepath(lang="nl"), header=True, delimiter=" ")

    # pad words with spaces to create transition probability between alphabets and spaces
    words = [' ' + w + ' ' for w in words]

    gb = GibberishDetectorClassifier()
    gb.fit(words)

    good_words = ['natuurlijk', 'fijnavond', 'smaakelijk', 'telefoon']
    bad_words = ['adaefgr', 'efsgtb rdrfw', 'afrvd telefoon', 'nietzogood']

    # Find the probability of generating a few arbitrarily chosen good and
    # bad phrases.
    good_probs = gb.predict_proba(good_words)
    bad_probs = gb.predict_proba(bad_words)
    print(good_probs, bad_probs)

    # Assert that we actually are capable of detecting the junk.
    assert min(good_probs) > max(bad_probs)

    # And pick a threshold halfway between the worst good and best bad inputs.
    thresh = (min(good_probs) + max(bad_probs)) / 2
    print(thresh)  # threshold = 0.0584
