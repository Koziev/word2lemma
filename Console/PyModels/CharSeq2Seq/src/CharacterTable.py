# -*- coding: utf-8 -*-
'''

(c) by Koziev Ilya inkoziev@gmail.com
'''

from __future__ import print_function
import numpy as np
import itertools


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen, sentinel_char, invert):
        # Нам нужно, чтобы символ-заполнитель гарантировано оказался первым в списке
        # и получил особый код 0, который будет исключен и backprop'а.
        self.chars = list( itertools.chain( sentinel_char, sorted(set(filter(lambda c:c!=sentinel_char,chars))) ) )
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen
        self.invert = invert
        self.sentinel_char = sentinel_char

    def is_good_word(self, word):
        for c in word:
            if not c in self.chars:
                return False;

        return True;

    def pad_word(self, word):
        res = word + self.sentinel_char * (self.maxlen - len(word))
        if self.invert:
            return res[::-1]
        else:
            return res

    def pad_lemma(self, lemma):
        return lemma + self.sentinel_char * (self.maxlen - len(lemma))

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

