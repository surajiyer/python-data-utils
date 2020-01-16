# coding: utf-8

"""
    description: frequently used or unique regex patterns
    author: Suraj Iyer
"""

__all__ = ['RegexPattern']


class RegexPattern:

    WholeWordOnly = lambda w: r'\b{}\b'.format(w)
    Linebreak = r'\r+|\n+'
    Number = WholeWordOnly(r"[0-9]+([.,][0-9]+)?")
    TwoOrMoreSpaces = r'\s{2,}'
    Email = WholeWordOnly(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    NL_pc4 = WholeWordOnly(r'[0-9]{4}\s?\w{2}')
    SingleCharacterWord = WholeWordOnly(r'\w')
    TimeOfDay = r"([0[0-9]|1[0-9]|2[0-3]):[0-5][0-9](\s?a[.]?m[.]?|p[.]?m[.]?|A[.]?M[.]?|P[.]?M[.]?)?"
    Unicode = r"[^\x00-\x7F]"
    Quotes = lambda x: r"(\"{0}\"|\'{0}\')".format(x)
    URL = r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?'
    SpecialCharacters = r'[`!@#*-+{}\[\]:;\'"|\\,<>\/]'
    NumbersWithSuffix = r'[0-9]+(st|th|nd|rd)'
    VersionNumber3N = r"\b(\d+\.)?(\d+\.)?(\*|\d+)\b"
    Copyright = r"\(c\)|®|©|™"
    ThreePlusRepeatingCharacters = r"([a-z])\1{2,}"
    ApostropheWords = r"[\w]+['][\w]+(['][\w]+)?"
    MD5 = r"[a-fA-F0-9]{32}"
    GoogleKeywordsFromURL = r'(?:(?<=q=|\+)([^$"+#&,]+)(?!.*q=))'
