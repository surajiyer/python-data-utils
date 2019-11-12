# coding: utf-8

"""
    description: Spellchecker based on Levenshtein edit distance.
    author: Suraj Iyer
    Original author: Peter Norvig
    URL: https://norvig.com/spell-correct.html
"""

__all__ = ['SimpleSpellCheck']

from python_data_utils.nlp import trie
from typing import Sequence


class SimpleSpellCheck:
    """
    Spell checker using basic Peter Norvig's edit distance with additional support for Trie-based word dictionaries.
    """

    @staticmethod
    def load_from_trie(filepath: str, file_type: str = "pkl"):
        file_type = file_type.lower()
        assert file_type in ('pkl', 'json', 'txt')

        spellcheck = SimpleSpellCheck()

        if file_type == "json":
            spellcheck.WORDS = trie.Trie().load_from_json(filepath)
        elif file_type == "pkl":
            spellcheck.WORDS = trie.Trie().load_from_pickle(filepath)
        elif file_type == "txt":
            spellcheck.WORDS = trie.Trie().load_from_text_corpus(filepath)
        spellcheck.N = spellcheck.WORDS.root['count']

        return spellcheck

    def P(self, word: str, N=None) -> float:
        """Probability of `word`."""
        assert all(getattr(self, attr, None) is not None for attr in ['N', 'WORDS']), ''
        if N is None:
            N = self.N
        return int(self.WORDS.get(word, 'count', 0)) / N

    # def known(self, words):
    #     """The subset of `words` that appear in the dictionary of WORDS."""
    #     return set(w for w in words if w in self.WORDS)

    def candidates(self, word: str, max_dist: int = 2) -> set:
        """Generate possible spelling corrections for word."""
        # return (self.known([word]) or self.known(utils.edit_dist(word, dist)) or [word])
        assert isinstance(max_dist, int) and max_dist > 0
        known = set()
        for dist in range(1, max_dist + 1):
            known |= self.WORDS.find_within_distance(word, dist)
        known |= set([word])
        return known

    def correct_word(
            self, word: str, dist: int = 2, error: str = 'ignore') -> str:
        """Most probable spelling correction for word."""
        candidates = self.candidates(word, dist)
        if error == 'ignore' and not candidates:
            return word
        elif error == 'raise' or not not candidates:
            return max(candidates, key=self.P)
        else:
            raise ValueError("'error' can only be 'raise' or 'ignore'")

    def correct_sentence(self, sentence: Sequence[str], dist: int = 2) -> str:
        return ' '.join(self.correct_word(w, dist) for w in sentence)
