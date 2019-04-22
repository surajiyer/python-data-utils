# coding: utf-8

"""
        description: Trie data structure
        URL: https://viblo.asia/p/nlp-build-a-trie-data-structure-from-scratch-with-python-3P0lPzroKox
        author: Suraj Iyer
"""

from collections.abc import MutableMapping
import dill
import json


class Trie(MutableMapping):
    # init Trie class
    def __init__(self):
        self.root = self.getNode()

        # the last added word node (can keep changing depending on last key added)
        self._prev_node = self.root

        # number of words in trie
        self.n_words = 0

    def getNode(self):
        return {"isEndOfWord": False, "children": {}}

    def add(self, word, additional_keys=None):
        assert isinstance(word, str), '{} must be a string'.format(word)
        assert isinstance(additional_keys, dict)

        current = self.root
        for ch in word:

            if ch in current["children"]:
                node = current["children"][ch]
            else:
                node = self.getNode()
                current["children"][ch] = node

            current = node

        current["isEndOfWord"] = True
        current['word'] = word

        # add additional info about words, e.g., count
        if additional_keys:
            current.update(additional_keys)

        # If word does not already exist
        if 'prev_node' not in current:
            # build a unidirectional linked-list between every consequent EOW node
            current['prev_node'] = self._prev_node
            self._prev_node['next_node'] = current
            self._prev_node = current

            # increment n_words
            self.n_words += 1

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

    def find(self, word):
        assert isinstance(word, str), '{} must be a string'.format(word)

        current = self.root
        for ch in word:
            if ch not in current["children"]:
                return False
            node = current["children"][ch]
            current = node

        return current["isEndOfWord"], {k: current[k] for k in current.keys() if k not in ('word', 'children', 'isEndOfWord', 'prev_node', 'next_node')}

    def get(self, word, additional_key=None, default=None):
        isEndOfWord, node = self.find(word)

        if isEndOfWord:
            if additional_key is None:
                return node
            elif additional_key not in node:
                if not default:
                    raise ValueError('{} does not contain key {}'.format(word, additional_key))
                else:
                    return default
            else:
                return node[additional_key]
        else:
            if not default:
                raise ValueError('{} not found'.format('word'))
            else:
                return default

    def findPrefix(self, word):
        assert isinstance(word, str), '{} must be a string'.format(word)

        current = self.root
        for ch in word:
            if ch not in current["children"]:
                return False
            node = current["children"][ch]
            current = node

        # return True if children contain keys and values
        return bool(current["children"])

    def find_within_distance(self, word, dist=2):
        assert isinstance(word, str), '{} must be a string'.format(word)
        from .utils import edit_dist
        return [word for word in edit_dist(word, dist) if self.find(word)[0]]

    def _delete(self, current, word, index):
        assert isinstance(word, str), '{} must be a string'.format(word)

        if(index == len(word)):
            if not current["isEndOfWord"]:
                return False
            current["isEndOfWord"] = False
            current['prev_node']['next_node'] = current['next_node']
            current['next_node']['prev_node'] = current['prev_node']
            return len(current["children"].keys()) == 0

        ch = word[index]
        if ch not in current["children"]:
            return False
        node = current["children"][ch]

        should_delete_current_node = self._delete(node, word, index + 1)

        if should_delete_current_node:
            current["children"].pop(ch)
            return len(current["children"].keys()) == 0

        return False

    def remove(self, word):
        self._delete(self.root, word, 0)
        self.n_words -= 1
        return self

    def removeAll(self, words):
        for word in words:
            self.remove(word)

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

    def __getitem__(self, key):
        if isinstance(key, str):
            key = key.split('__')
            if len(key) < 2:
                return self.get(key[0])
            else:
                val = self.get(key[0], key[1])
                key.pop(0)
                key.pop(0)
                for k in key:
                    val = val[k]
                return val
        else:
            return self.get(key)

    def __setitem__(self, key, value):
        self.add(key, value)

    def __delitem__(self, key):
        self.remove(key)

    def __iter__(self):
        current = self.root['next_node']
        while 'next_node' in current:
            yield current['word']
            current = current['next_node']
        yield current['word']

    def __len__(self):
        return self.n_words

    def __contains__(self, key):
        return self.find(key)
