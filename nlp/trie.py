# coding: utf-8

"""
        description: Trie data structure
        URL: https://viblo.asia/p/nlp-build-a-trie-data-structure-from-scratch-with-python-3P0lPzroKox
        author: Suraj Iyer
"""

import dill
import json


class Trie:
    # init Trie class
    def __init__(self):
        self.root = self.getNode()

    def getNode(self):
        return {"isEndOfWord": False, "children": {}}

    def add(self, word, additional_keys=None):
        current = self.root
        for ch in word:

            if ch in current["children"]:
                node = current["children"][ch]
            else:
                node = self.getNode()
                current["children"][ch] = node

            current = node
        current["isEndOfWord"] = True
        
        # add additional info about words, e.g., count
        if additional_keys:
            current.update(additional_keys)
        return self

    def addAll(self, words):
        try:
            word = next(words)
        except:
            return self
        if isinstance(word, dict):
            self.add(word.pop('word'), word)
            for word in words:
                self.add(word.pop('word'), word)
        elif isinstance(word, str):
            self.add(word)
            for word in words:
                self.add(word)
        else:
            raise ValueError('words format incorrect.')
        return self

    def find(self, word, additional_keys=None):
        if isinstance(additional_keys, str):
            additional_keys = [additional_keys]
        current = self.root
        for ch in word:
            if not (ch in current["children"]):
                return False
            node = current["children"][ch]
            current = node

        if not additional_keys or not current["isEndOfWord"]:
            return current["isEndOfWord"]
        else:
            return {arg: current[arg] for arg in set(additional_keys).intersection(current.keys())}

    def get(self, word, additional_key, default=None):
        x = self.find(word, additional_key)
        if x is False and default is None:
            raise ValueError('{} not found'.format('word'))
        elif default is None:
            raise ValueError('{} does not contain key {}'.format(word, additional_key))
        else:
            return default if x is False else x[additional_key]

    def findPrefix(self, word):
        current = self.root
        for ch in word:
            if not (ch in current["children"]):
                return False
            node = current["children"][ch]
            current = node

        # return True if children contain keys and values
        return bool(current["children"])

    def find_within_distance(self, word, dist=2):
        assert isinstance(word, str)
        from .utils import edit_dist
        return [word for word in edit_dist(word, dist) if self.find(word)]

    def remove(self, word):
        self._delete(self.root, word, 0)
        return self

    def _delete(self, current, word, index):
        if(index == len(word)):
            if not current["isEndOfWord"]:
                return False
            current["isEndOfWord"] = False
            return len(current["children"].keys()) == 0

        ch = word[index]
        if not (ch in current["children"]):
            return False
        node = current["children"][ch]

        should_delete_current_node = self._delete(node, word, index + 1)

        if should_delete_current_node:
            current["children"].pop(ch)
            return len(current["children"].keys()) == 0

        return False

    def save_to_pickle(self, file_name):
        f = open(file_name + ".pkl", "wb")
        dill.dump(self.root, f)
        f.close()

    def load_from_pickle(self, file_name):
        f = open(file_name + ".pkl", "rb")
        self.root = dill.load(f)
        f.close()
        return self

    def save_to_json(self, file_name):
        json_data = json.dumps(self.root)
        f = open(file_name + ".json", "w")
        f.write(json_data)
        f.close()

    def load_from_json(self, file_name):
        json_file = open(file_name + ".json", "r")
        self.root = json.load(json_file)
        json_file.close()
        return self

    def __contains__(self, key):
        return self.find(key)