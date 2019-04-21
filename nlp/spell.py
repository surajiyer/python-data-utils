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
    def __init__(self, dict_file_path, file_type="csv", delimiter=" "):
        # Read words from dictionary
        # self.WORDS = {line.split(' ')[0]: int(line.split(' ')[1]) for line in open(dict_file_path).readlines()}
        # self.N = sum(self.WORDS.values())
        file_type = file_type.lower()
        if dict_file_path is None or file_type == "csv":
            self.WORDS, self.N = utils.build_trie_from_dict_file(dict_file_path, header='include', delimiter=delimiter, 
                callback=lambda f: sum(int(line.split(delimiter)[1]) for line in f.readlines()))
        elif file_type == "json":
            self.WORDS = utils.Trie().load_from_json(dict_file_path)
            self.N = self.WORDS.root['count']
        elif file_type == "pkl" or file_type == "pickle":
            self.WORDS = utils.Trie().load_from_pickle(dict_file_path)
            self.N = self.WORDS.root['count']
        else:
            raise ValueError("Unsupported file type: {}".format(file_type))

    def P(self, word, N=None): 
        """Probability of `word`."""
        if N is None:
            N = self.N
        return int(self.WORDS.get(word, 'count', 0)) / N

    # def known(self, words):
    #     """The subset of `words` that appear in the dictionary of WORDS."""
    #     return set(w for w in words if w in self.WORDS)

    def candidates(self, word, dist=2): 
        """Generate possible spelling corrections for word."""
        # return (self.known([word]) or self.known(utils.edit_dist(word, dist)) or [word])
        return self.WORDS.find_within_distance(word, dist) + ([word] if self.WORDS.find(word) else [])

    def correct_word(self, word, dist=2, error='ignore'):
        """Most probable spelling correction for word."""
        candidates = self.candidates(word, dist)
        if error == 'ignore' and not candidates:
            return word
        elif error == 'raise' or not not candidates:
            return max(candidates, key=self.P)
        else:
            raise ValueError("'error' can only be 'raise' or 'ignore'")

    def correct_sentence(self, sentence, dist=2):
        return ' '.join(self.correct_word(w, dist) for w in sentence)
