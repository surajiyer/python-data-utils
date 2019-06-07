# coding: utf-8

"""
    description: Spellchecker based on Levenshtein edit distance.
    author: Suraj Iyer
    Original author: Peter Norvig
    URL: https://norvig.com/spell-correct.html
"""

from .. import utils


class SimpleSpellCheck:
    """
    Spell checker using basic Peter Norvig's edit distance with additional support for Trie-based word dictionaries.
    """

    @staticmethod
    def load_from_corpus(corpus, is_file_path):
        spellcheck = SimpleSpellCheck()
        if is_file_path:
            spellcheck.WORDS = utils.build_trie_from_corpus_file(corpus)
        else:
            spellcheck.WORDS = utils.build_trie_from_corpus(corpus)
        spellcheck.N = spellcheck.WORDS.root['count']
        return spellcheck

    @staticmethod
    def load_from_dictionary_csv(dictionary_file, delimiter=" "):
        spellcheck = SimpleSpellCheck()
        spellcheck.WORDS, spellcheck.N = utils.build_trie_from_dict_file(dictionary_file, header='include', delimiter=delimiter,
                                                                         callback=lambda f: sum(int(line.split(delimiter)[1]) for line in f.readlines()))
        spellcheck.WORDS.root['count'] = spellcheck.N

        # Update the counts to int type
        for word in spellcheck.WORDS:
            spellcheck.WORDS.add(
                word, {'count': int(spellcheck.WORDS['{}__count'.format(word)])}, update=True)

        return spellcheck

    @staticmethod
    def load_from_trie(dictionary_file, file_type="pkl"):
        file_type = file_type.lower()
        assert any(file_type == x for x in ('pkl', 'json'))

        spellcheck = SimpleSpellCheck()

        if file_type == "json":
            spellcheck.WORDS = utils.Trie().load_from_json(dictionary_file)
        elif file_type == "pkl" or file_type == "pickle":
            spellcheck.WORDS = utils.Trie().load_from_pickle(dictionary_file)
        spellcheck.N = spellcheck.WORDS.root['count']

        return spellcheck

    def P(self, word, N=None):
        """Probability of `word`."""
        assert all(getattr(self, attr, None) is not None for attr in ['N', 'WORDS']), ''
        if N is None:
            N = self.N
        return int(self.WORDS.get(word, 'count', 0)) / N

    # def known(self, words):
    #     """The subset of `words` that appear in the dictionary of WORDS."""
    #     return set(w for w in words if w in self.WORDS)

    def candidates(self, word, max_dist=2):
        """Generate possible spelling corrections for word."""
        # return (self.known([word]) or self.known(utils.edit_dist(word, dist)) or [word])
        assert isinstance(max_dist, int) and max_dist > 0
        known = set()
        for dist in range(1, max_dist + 1):
            known |= self.WORDS.find_within_distance(word, dist)
        known |= set([word])
        return known

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