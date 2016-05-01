# -*- coding: utf-8 -*-
'''
Использование архитектуры sequence 2 sequence для выполнения лемматизации русских слов.
В качестве обучающего набора используется выборка из SQL словаря русского языка http://www.solarix.ru/sql-dictionary-sdk.shtml
Небольшая выгрузка объемом 100k включена в релиз и может быть использована для начального тестирования. Для
получения полного набора в 3 млн паттернов используйте SQL словарь.

По мере обучения модель делает пробные лемматизации, часть которых печатает на экране, и сохраняет
в тестовом файле 'word2lemma.results.txt'.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Masking
import keras.callbacks
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
from six.moves import range


# Parameters for the model and dataset
SENTINEL_CHAR = u' '
TRAINING_SIZE = 3000000
INVERT = True
RNN = recurrent.LSTM
HIDDEN_SIZE = 64
BATCH_SIZE = 128
LAYERS = 1


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

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


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

class VizualizeCallback(keras.callbacks.Callback):

    def __init__(self):
        self.epoch = 0
        self.output_path = 'word2lemma.results.txt'
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1
        # Select N samples from the validation set at random so we can visualize errors
        with open( self.output_path, 'a' ) as fsamples:
            fsamples.write( u'\n' + '='*50 + u'\nepoch=' + str(self.epoch) + u'\n' );
            for i in range(100):
                ind = np.random.randint(0, len(X_test))
                rowX, rowy = X_test[np.array([ind])], y_test[np.array([ind])]
                preds = model.predict_classes(rowX, verbose=0)
                q = ctable.decode(rowX[0])
                wordform = q[::-1] if INVERT else q
                correct = ctable.decode(rowy[0])
                guess = ctable.decode(preds[0], calc_argmax=False)
                if i<10:
                    print('Wordform:', wordform )
                    print('Lemma:  ', correct)
                    print(colors.ok + '☑ ' + colors.close if correct == guess else colors.fail + '☒ ' + colors.close, guess)
                    print('---')
                
                fsamples.write( (wordform + u' ==> ' + guess + u'\n').encode('utf-8') )



# ----------------------------------------------------------------------


corpus_path = 'word2lemmas.dat'

print('Loading data', corpus_path, '...')

word2lemmas = []

max_word_len = 0
max_lemma_len = 0

chars = set([SENTINEL_CHAR])

with open( corpus_path, 'r' ) as fdata:

    fdata.readline()

    while len(word2lemmas) < TRAINING_SIZE:

        toks = fdata.readline().strip().decode('utf-8').split('\t')

        word = toks[0]
        lemma = toks[1]

        if ' ' not in word and ' ' not in lemma:
            max_word_len = max( max_word_len, len(word) )
            max_lemma_len = max( max_lemma_len, len(lemma) )
            word2lemmas.append( (word,lemma) )
            chars.update( list(word) )
            chars.update( list(lemma) )

ctable = CharacterTable(chars, max_word_len)

print('Total word2lemma patterns:', len(word2lemmas))
print('max_word_len=', max_word_len );
print('max_lemma_len=', max_lemma_len );

questions = []
expected = []

for ipattern,word2lemma in enumerate(word2lemmas):
    
    # Pad the data with spaces such that it is always MAXLEN
    q = word2lemma[0]
    query = q + SENTINEL_CHAR * (max_word_len - len(q))
    if INVERT:
        query = query[::-1]
    
    a = word2lemma[1]
    answer = a + SENTINEL_CHAR *( max_lemma_len-len(a))
    
    questions.append(query)
    expected.append(answer)


n_patterns = len(questions)
test_share = 0.1
n_test = int(n_patterns*test_share)
n_train = n_patterns-n_test

bits_per_char = len(chars)

print('Vectorization...')
X_train = np.zeros((n_train, max_word_len, bits_per_char), dtype=np.bool)
y_train = np.zeros((n_train, max_lemma_len, bits_per_char), dtype=np.bool)

X_test = np.zeros((n_test, max_word_len, bits_per_char), dtype=np.bool)
y_test = np.zeros((n_test, max_lemma_len, bits_per_char), dtype=np.bool)

i_test = 0
i_train = 0
for i in range(len(questions)):

    word = questions[i]
    lemma = expected[i]

    if i<n_test:
        X_test[i_test] = ctable.encode(word, maxlen=max_word_len)
        y_test[i_test] = ctable.encode(lemma, maxlen=max_lemma_len)
        i_test = i_test+1
    else:
        X_train[i_train] = ctable.encode(word, maxlen=max_word_len)
        y_train[i_train] = ctable.encode(lemma, maxlen=max_lemma_len)
        i_train = i_train+1
        
print('Build model...')
model = Sequential()
model.add( Masking( mask_value=0,input_shape=(max_word_len,bits_per_char) ) )
model.add( RNN( HIDDEN_SIZE, input_shape=(max_word_len, bits_per_char) ) )
model.add( RepeatVector(max_lemma_len) )
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributedDense(bits_per_char))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


model_checkpoint = ModelCheckpoint( 'word2lemma.model', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping( monitor='val_loss', patience=5, verbose=1, mode='auto')

vizualizer = VizualizeCallback()

model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=30, validation_data=(X_test, y_test), callbacks=[model_checkpoint,early_stopping,vizualizer] )

