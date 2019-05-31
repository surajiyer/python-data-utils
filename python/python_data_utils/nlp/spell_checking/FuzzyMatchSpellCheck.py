# coding: utf-8

"""
    description: Spell check with Fuzzy matching between input and dictionary words.
    author: Suraj Iyer
    Original author: pragnakalp@github
    URL: https://github.com/pragnakalp/spellcheck-using-dictionary-in-python.
"""

from fuzzywuzzy import fuzz


class FuzzyMatchSpellCheck:

    # initialization method
    def __init__(self, words_list):
        # store all the words into a class variable dictionary
        self.dictionary = words_list

    # this method returns the possible suggestions of the correct words
    def suggestions(self, text, delimiter=" ", threshold=75):
        # store the words of the string to be checked in a list by using a split function
        string_words = text.split(delimiter)

        # a list to store all the possible suggestions
        suggestions = []

        # loop over the number of words in the string to be checked
        for i in range(len(string_words)):

            # loop over words in the dictionary
            for name in self.dictionary:

                # if the fuzzywuzzy returns the matched value greater than threshold
                if fuzz.ratio(string_words[i].lower(), name.lower()) >= threshold:

                    # append the dict word to the suggestion list
                    suggestions.append(name)

        # return the suggestions list
        return suggestions

    # this method returns the corrected string of the given input
    def correct(self, text, delimiter=" ", threshold=75):
        # store the words of the string to be checked in a list by using a split function
        string_words = text.split(delimiter)

        # loop over the number of words in the string to be checked
        for i in range(len(string_words)):

            # initialize a maximum probability variable to 0
            max_percent = 0

            # loop over the words in the dictionary
            for name in self.dictionary:

                # calculate the match probability
                percent = fuzz.ratio(string_words[i].lower(), name.lower())

                # if fuzzywuzzy returns the matched value greater than threshold
                if percent >= threshold:

                    # if the matched probability is
                    if percent > max_percent:

                        # change the original value with the corrected matched value
                        string_words[i] = name

                    # change the max percent to the current matched percent
                    max_percent = percent

        # return the corrected string
        return delimiter.join(string_words)
