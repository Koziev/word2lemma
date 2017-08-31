# -*- coding: utf-8 -*-
'''

Модель лемматизатора должна быть обучена и сохранена на диск - см. word2lemma_train.py

(c) by Koziev Ilya inkoziev@gmail.com
'''

from __future__ import print_function
from keras.models import Sequential
from keras.models import model_from_json
import numpy as np
import json
import codecs
import pickle
import os
import sys

class Lemmatizer(object):
    def __init__(self):
        pass

    def load(self, data_folder):
        with open(os.path.join(data_folder,'ctable.dat'), 'r') as f:
            self.ctable = pickle.load(f)

        with open(os.path.join(data_folder,'word2lemma.arch'),'r') as f:
            self.model = model_from_json(f.read())

        self.model.load_weights( os.path.join(data_folder,'word2lemma.model') )

    def predict(self, word):
        padded_word = self.ctable.pad_word(word)

        X_query = np.zeros((1, self.ctable.maxlen, len(self.ctable.chars)), dtype=np.bool)
        X_query[0] = self.ctable.encode(padded_word)

        r = self.model.predict_classes(X_query, verbose=0)
        lemma = self.ctable.decode(r[0], calc_argmax=False).strip()
        return lemma
