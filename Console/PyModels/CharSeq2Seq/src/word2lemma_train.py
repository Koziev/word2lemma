# -*- coding: utf-8 -*-
'''
Использование архитектуры sequence 2 sequence для выполнения лемматизации русских слов.
В качестве обучающего набора используется выборка из SQL словаря русского языка http://www.solarix.ru/sql-dictionary-sdk.shtml
Небольшая выгрузка объемом 100k включена в релиз и может быть использована для начального тестирования.
Для получения полного набора в 3 млн паттернов используйте SQL словарь.

По мере обучения модель делает пробные лемматизации, часть которых печатает на экране, и сохраняет
в тестовом файле 'word2lemma.results.txt'.

В результате работы данного скрипта получаем файл с весами нейросетки и информацией по символам.
Далее эти файлы используются в тестовой программе для лемматизации вводимых с консоли слов.

См. также: "Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

(c) by Koziev Ilya inkoziev@gmail.com  https://github.com/Koziev/word2lemma
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Activation, RepeatVector, Dense, Masking
from keras.layers.wrappers import TimeDistributed
import keras.callbacks
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import codecs
import itertools
import pickle
from CharacterTable import CharacterTable


# Model and dataset parameters
SENTINEL_CHAR = u' '
TRAINING_SIZE = 3000000
INVERT = True
HIDDEN_SIZE = 64
BATCH_SIZE = 256
NB_EPOCHS = 50
LAYERS = 1

corpus_path = '../../../../data/word2lemmas.dat'
data_folder = '../../../../data'

# ------------------------------------------------------------------------

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

class VisualizeCallback(keras.callbacks.Callback):

    def __init__(self, X_test, y_test, model, ctable):
        self.epoch = 0
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.ctable = ctable
        self.output_path = '../../../../data/word2lemma.results.txt'
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1
        # Select N samples from the validation set at random so we can visualize errors
        with open( self.output_path, 'a' ) as fsamples:
            fsamples.write( u'\n' + '='*50 + u'\nepoch=' + str(self.epoch) + u'\n' );
            for i in range(100):
                ind = np.random.randint(0, len(self.X_test))
                rowX, rowy = self.X_test[np.array([ind])], self.y_test[np.array([ind])]
                preds = self.model.predict_classes(rowX, verbose=0)
                q = self.ctable.decode(rowX[0])
                wordform = q[::-1] if INVERT else q
                correct = self.ctable.decode(rowy[0])
                guess = self.ctable.decode(preds[0], calc_argmax=False)
                if i<10:
                    print(colors.ok + '☑ ' + colors.close if correct == guess else colors.fail + '☒ ' + colors.close, end='')
                    print(u'wordform={} true_lemma={} model_lemma={}'.format(wordform, correct, guess) )

                fsamples.write( (wordform + u' ==> ' + guess + u'\n').encode('utf-8') )

# ----------------------------------------------------------------------

class LemmatizerTrainer(object):
    def __init__(self):
        pass

    def fit(self, corpus_path, data_folder, nb_samples):
        print(u'Loading data from {}...'.format(corpus_path))

        word2lemmas = []
        max_len = 0
        chars = set([SENTINEL_CHAR])

        with codecs.open( corpus_path, 'r', 'utf-8' ) as fdata:

            fdata.readline()

            while len(word2lemmas) < nb_samples:

                toks = fdata.readline().strip().split(u'\t')

                word = toks[0]
                lemma = toks[1]

                if u' ' not in word and u' ' not in lemma:
                    max_len = max( max_len, len(word), len(lemma) )
                    word2lemmas.append( (word, lemma) )
                    chars.update( itertools.chain(word,lemma) )


        ctable = CharacterTable(chars, max_len, SENTINEL_CHAR, INVERT)

        with open(os.path.join(data_folder,'ctable.dat'), 'w') as f:
            pickle.dump(ctable, f)

        print('number of samples={}'.format(len(word2lemmas)))
        print('max_len={}'.format(max_len));

        questions = [ ctable.pad_word(word2lemma[0]) for word2lemma in word2lemmas ]
        expected = [ ctable.pad_lemma(word2lemma[1]) for word2lemma in word2lemmas ]

        n_patterns = len(questions)
        test_share = 0.1
        n_test = int(n_patterns*test_share)
        n_train = n_patterns-n_test

        bits_per_char = len(chars)

        print('Vectorization...')
        X_train = np.zeros((n_train, max_len, bits_per_char), dtype=np.bool)
        y_train = np.zeros((n_train, max_len, bits_per_char), dtype=np.bool)

        X_test = np.zeros((n_test, max_len, bits_per_char), dtype=np.bool)
        y_test = np.zeros((n_test, max_len, bits_per_char), dtype=np.bool)

        i_test = 0
        i_train = 0
        for i in range(len(questions)):

            word = questions[i]
            lemma = expected[i]

            if i<n_test:
                X_test[i_test] = ctable.encode(word, maxlen=max_len)
                y_test[i_test] = ctable.encode(lemma, maxlen=max_len)
                i_test = i_test+1
            else:
                X_train[i_train] = ctable.encode(word, maxlen=max_len)
                y_train[i_train] = ctable.encode(lemma, maxlen=max_len)
                i_train = i_train+1

        print('Build model...')
        model = Sequential()
        model.add( Masking( mask_value=0, input_shape=(max_len, bits_per_char) ) )
        model.add( recurrent.LSTM( HIDDEN_SIZE, input_shape=(max_len, bits_per_char) ) )
        model.add( RepeatVector(max_len) )
        for _ in range(LAYERS):
            model.add(recurrent.LSTM(HIDDEN_SIZE, return_sequences=True))

        model.add( TimeDistributed(Dense(bits_per_char)) )
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        with open(os.path.join(data_folder,'word2lemma.arch'),'w') as f:
            f.write(model.to_json())

        model_checkpoint = ModelCheckpoint( os.path.join(data_folder,'word2lemma.model'),
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='auto')

        early_stopping = EarlyStopping( monitor='val_loss',
                                        patience=10,
                                        verbose=1,
                                        mode='auto')

        visualizer = VisualizeCallback(X_test, y_test, model, ctable)

        model.fit(X_train,
                  y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NB_EPOCHS,
                  validation_data=(X_test, y_test),
                  callbacks=[model_checkpoint, early_stopping, visualizer] )

# ----------------------------------------------------------------------

trainer = LemmatizerTrainer()
trainer.fit(corpus_path, data_folder, TRAINING_SIZE)
