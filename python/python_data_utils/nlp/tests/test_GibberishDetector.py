from ..GibberishDetector import GibberishDetectorClassifier
from .. import utils


def test_1():
    # load words
    words = utils.words_set_dictionary('dutch_dictionary_small')

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
