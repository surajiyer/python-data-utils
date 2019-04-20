# coding: utf-8

"""
    description: Spellchecker based on Levenshtein edit distance.
    author: Suraj Iyer
    Original author: Peter Norvig
    URL: https://norvig.com/spell-correct.html
"""

from os.path import dirname, join
from . import utils


class SpellCheck:
    """
    Spell checker
    """
    def __init__(self, file_path=None, delimiter=" "):
        # Read words from dictionary
        if file_path is None:
            file_path = join(dirname(__file__), 'en', '{}_{}.txt'.format('en', '50k'))
        # self.WORDS = {line.split(' ')[0]: int(line.split(' ')[1]) for line in open(file_path).readlines()}
        # self.N = sum(self.WORDS.values())
        self.WORDS, self.N = utils.build_trie_from_dict_file(file_path, header='include', delimiter=delimiter, 
            cb=lambda f: sum(int(line.split(delimiter)[1]) for line in f.readlines()))

    def P(self, word, N=None): 
        """Probability of `word`."""
        if N is None:
            N = self.N
        return self.WORDS.get(word, 0) / N

    # def known(self, words):
    #     """The subset of `words` that appear in the dictionary of WORDS."""
    #     return set(w for w in words if w in self.WORDS)

    def candidates(self, word, dist=2): 
        """Generate possible spelling corrections for word."""
        # return (self.known([word]) or self.known(utils.edit_dist(word, dist)) or [word])
        return self.WORDS.find_within_distance(word, dist) + ([word] if self.WORDS.find(word) else [])

    def correct_word(self, word):
        """Most probable spelling correction for word."""
        return max(self.candidates(word), key=self.P)

    def correct_sentence(self, sentence):
        return ' '.join(self.correct_word(w) for w in sentence)
