# coding: utf-8

"""
    description: Spellchecker based on Levenshtein edit distance.
    author: Suraj Iyer
    Original author: Peter Norvig (https://norvig.com/spell-correct.html)
"""

from os.path import dirname, join


class SpellCheck:
    """
    Spellchecker

    lang: str, default='en'
        Currently only English (en) / Dutch (nl) supported.
    dictionary_size: str, default='50k'
        Use the small dictionary containing only top '50k' words or 'full' dictionary.
        Full dictionary is very large and can result in large running times.
    """
    def __init__(self, lang='en', dictionary_size='50k'):
        # Read words from dictionary
        filepath = join(dirname(__file__), lang, '{}_{}.txt'.format(lang, dictionary_size))
        self.WORDS = {line.split(' ')[0]: int(line.split(' ')[1]) for line in open(filepath).readlines()}
        self.N = sum(self.WORDS.values())

    def P(self, word, N=None): 
        """Probability of `word`."""
        if N is None:
            N = self.N
        words = word.split(' ')
        if len(words) > 1:
            return sum(self.WORDS.get(w, 0) for w in words) / N
        return self.WORDS.get(word, 0) / N

    def correct_word(self, word): 
        """Most probable spelling correction for word."""
        return max(self.candidates(word), key=self.P)

    def correct_sentence(self, sentence):
        return ' '.join(self.correct_word(w) for w in sentence)

    def known(self, words): 
        """The subset of `words` that appear in the dictionary of WORDS."""
        return set(w for w in words if w in self.WORDS)

    def candidates(self, word, edit_distance=2): 
        """Generate possible spelling corrections for word."""
        return (self.known([word]) or self.known(SpellCheck.edits_n(word, edit_distance)) or [word])

    @staticmethod
    def edits_1(word):
        """All edits that are one edit away from `word`."""
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    @staticmethod
    def edits_2(word):
        """All edits that are two edits away from `word`."""
        return (e2 for e1 in SpellCheck.edits_1(word) for e2 in SpellCheck.edits_1(e1))

    @staticmethod
    def edits_n(word, n=2, k=None):
        """All edits that are `n` edits away from `word`."""
        if k is None:
            k = n
        assert n > 0 and k > 0
        if k == 1:
            return SpellCheck.edits_1(word)
        new_words = (w2 for w1 in SpellCheck.edits_1(word) for w2 in SpellCheck.edits_n(w1, n, k-1))
        return list(set(new_words)) if k == n else new_words
