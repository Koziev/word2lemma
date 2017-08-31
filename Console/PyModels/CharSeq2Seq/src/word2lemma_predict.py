# -*- coding: utf-8 -*-
'''

Консольный лемматизатор: использование обученной модели сеточного лемматизатора
для нормализации вводимых с клавиатуры слов.

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
from Lemmatizer import Lemmatizer

MODEL_FOLDER = '../../../../data'

# ----------------------------------------------------------------------

lemmatizer = Lemmatizer()
lemmatizer.load(MODEL_FOLDER)

# теперь ввод слова в консоли и печать результата лемматизации
while True:
    word = raw_input('\n>: ').strip().decode(sys.stdout.encoding)

    if len(word) == 0:
        break;

    lemma = lemmatizer.predict(word)
    print(u'lemma={0}'.format(lemma))





















