# -*- coding: utf-8 -*-
'''

Модели для лемматизации слов.

char-rnn:
--------

Использование архитектуры sequence 2 sequence для обучения модели, которая
для любого слова строит вектор фиксированной длины, а затем из этого
вектора разворачивает лемму слова.

char-feedforward:
----------------

Использование feed-forward сетки для построения автоэнкодера, который преобразует
цепочку символов в короткий вещественный вектор.

'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Masking
from keras.layers.core import Dense
import keras.callbacks
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
import numpy as np
import os
from six.moves import range
import random
import json
from collections import Counter
from collections import defaultdict
from gensim.models import word2vec
import codecs
import pickle


WORD2LEMMA_FILE = 'word2lemma.dat'


# Parameters for the model and dataset
SENTINEL_CHAR = u' '
TRAINING_SIZE = 5000000
INVERT = True
RNN = recurrent.LSTM
#RNN = recurrent.GRU
FF_ACTIVATION = 'sigmoid'
DROPOUT = 0.0
HIDDEN_SIZE = 64
BATCH_SIZE = 128
P_MISSPELLING = 0.0 # вероятность генерации опечаток для словоформы
N_MISSPELLINGS_PER_WORD = 2 # кол-во вариантов с опечатками на одну правильную словоформу
N_EPOCH0 = 1000

DATASET_CONFIG = 'dataset.config'

CHAR_RNN_WEIGHTS_FILENAME = 'char_rnn.#.model'
CHAR_RNN_ARCH_FILENAME    = 'char_rnn.arch'
CHAR_RNN_HISTORY_FILENAME = 'char_rnn.#.history.dat'

CHAR_FEEDFORWARD_WEIGHTS_FILENAME = 'char_feedforward.#.model'
CHAR_FEEDFORWARD_HISTORY_FILENAME = 'char_feedforward.#.history.dat'
CHAR_FEEDFORWARD_ARCH_FILENAME    = 'char_feedforward.arch'


# --------------------------------------------------------------------

class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        
    def is_good_word( self, word ):
        for c in word:
            if not c in self.chars:
                return False;

        return True;
        
    # заполнение тензора для rnn сетки
    def encode(self, C, maxlen):
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    # заполнение вектора для feedforward сетки
    def encode_feedforward(self, C, maxlen):
        nbit_per_char = len(self.chars)
        X = np.zeros( maxlen*nbit_per_char )
        for i, c in enumerate(C):
            X[ i*nbit_per_char + self.char_indices[c] ] = 1
        return X


    def decode( self, X, calc_argmax=True ):
        if calc_argmax:
            X = X.argmax(axis=-1)
           
        return ''.join(self.indices_char[x] for x in X)

    def decode_feedforward( self, X, calc_argmax ):
        nbit_per_char = len(self.chars)
        xlen = X.shape[0] / nbit_per_char # длина в символах
        X_rnn = np.zeros( (xlen, nbit_per_char) )
        for i in range(xlen):
            X_rnn[i] = X[ i*nbit_per_char : (i+1)*nbit_per_char ]
        
        return self.decode( X_rnn, calc_argmax )

# ----------------------------------------------------------------------

def v_cosine( a, b ):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


class VizualizeCallback_RNN1(keras.callbacks.Callback):

    def __init__(self,pos_name):
        self.epoch = 0
        self.pos_name = pos_name
        self.output_path = 'char_rnn.'+pos_name+'.quality.dat'
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1
        n_sample=0
        n_error=0
        # Select N samples at random so we can visualize errors
        with open( self.output_path, 'a' ) as fsamples:
            #fsamples.write( u'\n' + '='*50 + u'\nepoch=' + str(self.epoch) + u'\n' );
            for i in range(1000):
                ind = np.random.randint(0, len(X_train))
                rowX, rowy = X_train[np.array([ind])], y_train[np.array([ind])]
                preds = model.predict_classes(rowX, verbose=0)
                q = ctable.decode(rowX[0])
                wordform = q[::-1] if INVERT else q
                correct = ctable.decode(rowy[0])
                guess = ctable.decode(preds[0], calc_argmax=False)
                if i<10:
                    if correct == guess:
                        print( colors.ok + '☑ ' + colors.close, wordform, '==>', guess )
                    else:
                        print( colors.fail + '☒ ' + colors.close, wordform, '==>', guess, ' required=', correct )
                
                n_sample += 1
                if correct != guess:
                    n_error += 1

                #fsamples.write( (wordform + u' ==> ' + guess + u'\n').encode('utf-8') )
            fsamples.write( '{0}\t{1}\t{2}\n'.format( self.epoch, n_error, float(n_error)/float(n_sample) ) )
            fsamples.flush()
            print( 'n_error={0} error_rate={1}'.format( n_error, float(n_error)/float(n_sample) ) )
            
            print( 'DEBUG...' )
            left_word = u'кошка'
            X_query = np.zeros( (1, max_word_len, len(ctable.chars) ), dtype=np.bool)
            X_query[0] = ctable.encode( left_word, maxlen=max_word_len)

            r = model.predict_classes(X_query, verbose=0)
            lemma = ctable.decode(r[0], calc_argmax=False)
            print( u'DEBUG lemma={0}'.format(lemma) )
            
            
            

class VizualizeCallback_FEEDFORWARD1(keras.callbacks.Callback):

    def __init__(self,pos_name):
        self.epoch = 0
        self.pos_name = pos_name
        self.output_path = 'char_feedforward.'+pos_name+'.quality.dat'
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1
        
        n_sample=0
        n_error=0
        # Select N samples at random so we can visualize errors
        with open( self.output_path, 'a' ) as fsamples:
            #fsamples.write( u'\n' + '='*50 + u'\nepoch=' + str(self.epoch) + u'\n' );
            for i in range(10000):
                ind = np.random.randint(0, len(X_train))
                rowX, rowy = X_train[np.array([ind])], y_train[np.array([ind])]
                preds = model.predict(rowX, verbose=0)
                q = ctable.decode_feedforward(rowX[0], calc_argmax=True)
                wordform = q[::-1] if INVERT else q
                correct = ctable.decode_feedforward(rowy[0], calc_argmax=True)
                guess = ctable.decode_feedforward(preds[0], calc_argmax=True)

                if i<10:
                    if correct == guess:
                        print( colors.ok + '☑ ' + colors.close, wordform, '==>', guess )
                    else:
                        print( colors.fail + '☒ ' + colors.close, wordform, '==>', guess, ' required=', correct )

                n_sample += 1
                if correct != guess:
                    n_error += 1
                
                #fsamples.write( (wordform + u' ==> ' + guess + u'\n').encode('utf-8') )
            fsamples.write( '{0}\t{1}\t{2}\n'.format( self.epoch, n_error, float(n_error)/float(n_sample) ) )
            fsamples.flush()
            print( 'n_error={0} error_rate={1}'.format( n_error, float(n_error)/float(n_sample) ) )


# ----------------------------------------------------------------------

chars = set()


def prepare_chars():
    global chars
    chars = { c for c in u'абвгдежзийклмнопрстуфхцчшщъыьэюя' }
    chars.add( SENTINEL_CHAR )
    print( 'chars.count=', len(chars) )



# Вернет True, если все символы в слове word валидны
def all_chars_known( word ):
    for c in word:
        if not c in chars:
            return False
    return True        

# ----------------------------------------------------------------------

class Dataset:
    def __init__(self):
        self.max_word_len = -1
        self.max_lemma_len = -1
        self.class2word2lemma = defaultdict( lambda:[] )
        self.changing_classes = []
        
    def load(self):
        # из датасета загружаем три столбца СЛОВО \t ЛЕММА \t ЧАСТЬ_РЕЧИ
        # пары СЛОВО-ЛЕММА разпределяем по частям речи, чтобы для каждой части речи
        # строить отдельную модель.
        print( 'Analysing dataset {0}...'.format(WORD2LEMMA_FILE) )
        all_classes = set()
      
        with codecs.open( WORD2LEMMA_FILE, "r", "utf-8") as rdr:
            for line in rdr:
                px = line.strip().split('\t')
                word = px[0]
                lemma = px[1]
                p_o_s = px[2]
                
                if all_chars_known(word) and all_chars_known(lemma):
                    self.class2word2lemma[p_o_s].append( (word,lemma) )
                    self.max_word_len = max( self.max_word_len, len(word) )
                    self.max_lemma_len = max( self.max_lemma_len, len(lemma) )
                    all_classes.add( p_o_s )

        print( 'number of classes={0}'.format( len(self.class2word2lemma) ) )

        # оставим только части речи со словоизменением.
        for p_o_s in all_classes:
            if len(self.class2word2lemma[p_o_s])>1000: # берем только части речи с большим кол-вом пар слово-лемма, чтобы исключить предлоги (С-СО, ПОД-ПОДО,...)
                wordchange_detected = False
                lemmas = set()
                for (w,l) in self.class2word2lemma[p_o_s]:
                    if l in lemmas:
                        wordchange_detected = True
                        break
                    else:
                        lemmas.add(l)
                        
                if wordchange_detected:
                    self.changing_classes.append( p_o_s )

        # добавим также объединенный список пар слово-лемма для всех частей речи со словоизменением.
        all_word2lemma = []
        for p_o_s in self.changing_classes:
            all_word2lemma += self.class2word2lemma[p_o_s]

        self.class2word2lemma['ALL'] = all_word2lemma
        self.changing_classes.append( 'ALL' )

        # подготовка датасетов закончена, покажем статистику
        print( u'{0} changing_classes: {1}'.format( len(self.changing_classes), unicode.join( u' ', self.changing_classes ) ) )
        self.bits_per_char = len(chars)
        self.ctable = CharacterTable(chars)
        print('bits_per_char=', self.bits_per_char )
        print('max_word_len=', self.max_word_len )
        print('max_lemma_len=', self.max_lemma_len )        

    def get_word2lemmas(self,p_o_s):
        return self.class2word2lemma[p_o_s]

# ----------------------------------------------------------------------

prepare_chars()

while True:
    print( '\nChoose:' )
    print( '1 - train char-rnn [word --> lemma] autoencoder')
    print( '2 - train char-feedforward [word --> lemma] autoencoder' )
    print( '' )
    print( '0 - exit' )

    cmd = raw_input('Choose:> ').strip()

    if cmd=='0':
        break;

    if cmd=='1' or cmd=='2':

        model1_type = None
        if cmd=='1':
            model1_type = 'char-rnn'
        elif cmd=='2':
            model1_type = 'char-feedforward'
        else:
            raise NameError('not implemented')       
        
        print( 'Training ', model1_type, ' model' )
        print( 'HIDDEN_SIZE=', HIDDEN_SIZE )
        print( 'P_MISSPELLING=', P_MISSPELLING )

        dataset = Dataset()
        dataset.load()
        
        bits_per_char = dataset.bits_per_char
        ctable = dataset.ctable
        with open( 'ctable.dat', 'wt' ) as pfile:
            pickle.dump( ctable, pfile )
        
        max_word_len = dataset.max_word_len
        max_lemma_len = dataset.max_lemma_len
       
        # для каждой части речи строим отдельную модель
        #for p_o_s in dataset.changing_classes:
        for p_o_s in [u'ALL']:
            print( 'Working with word-lemma pairs of', p_o_s, 'part of speech' );
            
            word2lemma = dataset.get_word2lemmas( p_o_s )
            right_len = max_lemma_len
            
            with open(DATASET_CONFIG,'w') as cfg:
                params = { 
                          'model1_type':model1_type,
                          'max_word_len':max_word_len,
                          'max_lemma_len':max_lemma_len,
                          'INVERT':INVERT,
                          'HIDDEN_SIZE':HIDDEN_SIZE,
                          'FF_ACTIVATION':FF_ACTIVATION
                         }
                json.dump( params, cfg )
            
            # генерируем список фактически используемых строк на входе и на выходе модели,
            # с учетом возможного добавления опечаток.
            
            left_words = []
            right_words = []
            
            misspellings_count = 0

            char_replacement = {}
            cx = u'оашщеиьъдтсзбпхк'
            for i in range( 0, len(cx), 2 ):
                char_replacement[ cx[i] ] = cx[i+1]
                
            pattern_replacement = {}
            pattern_replacement[u'тся'] = u'ться'
            pattern_replacement[u'ться'] = u'тся'
            pattern_replacement[u'жи'] = u'жы'
            pattern_replacement[u'жы'] = u'жи'
            pattern_replacement[u'ши'] = u'шы'
            pattern_replacement[u'шы'] = u'ши'
            pattern_replacement[u'ца'] = u'тся'
            pattern_replacement[u'ца'] = u'тса'
            pattern_replacement[u'тса'] = u'ца'
            pattern_replacement[u'ча'] = u'чя'
            pattern_replacement[u'ща'] = u'щя'
            pattern_replacement[u'сч'] = u'щ'
            pattern_replacement[u'съе'] = u'се'
            pattern_replacement[u'съе'] = u'се'
            pattern_replacement[u'въе'] = u'ве'
            pattern_replacement[u'ве'] = u'въе'
            pattern_replacement[u'ого'] = u'ова'
            pattern_replacement[u'стн'] = u'сн'
            pattern_replacement[u'нн'] = u'н'
            pattern_replacement[u'сс'] = u'с'
            pattern_replacement[u'рр'] = u'р'
            pattern_replacement[u'дт'] = u'тт'


            for (word,lemma) in word2lemma:
                
                if len(left_words)>TRAINING_SIZE:
                    break

                left_word_len = len(word)

                # Pad the data with spaces such that it is always MAXLEN
                padded_word = word + SENTINEL_CHAR * (max_word_len - left_word_len)
                left_word = padded_word
                if INVERT:
                    left_word = left_word[::-1]
                
                left_words.append(left_word)

                right_word = lemma
                right_word_len = len(right_word)    
                right_word = right_word + SENTINEL_CHAR * (max_lemma_len - right_word_len)
                right_words.append(right_word)
                
                if P_MISSPELLING>0 and random.random()<=P_MISSPELLING:
                    for i in range(N_MISSPELLINGS_PER_WORD):
                        word2 = u''
                        scenario = random.randint(0, 4)
                        
                        if scenario==0:
                            # удваиваем любую букву
                            ichar = random.randint(0,left_word_len-1)
                            ch = word[ichar]
                            if ichar==0:
                                word2 = ch+word # удваиваем первый символ
                            elif ichar==left_word_len-1:
                                word2 = word+ch # удваиваем последний символ
                            else:
                                word2 = word[:ichar+1] + ch + word[ichar+1:] # удваиваем символ внутри слова
                            
                        elif scenario==1:
                            # удаляем любую букву
                            ichar = random.randint(0,left_word_len-1)
                            if ichar==0:
                                word2 = word[1:] # удаляем первый символ
                            elif ichar==left_word_len-1:
                                word2 = word[:left_word_len-1] # удаляем последний символ
                            else:
                                word2 = word[:ichar] + word[ichar+1:] # удаляем символ внутри слова
                           
                        elif scenario==2:
                            # замены букв
                            replacement_count=0
                            for ch in word:
                                
                                ch2 = ch
                                
                                if replacement_count==0:
                                    if ch in char_replacement:
                                        ch2 = char_replacement[ch]
                                        
                                if ch!=ch2:
                                    replacement_count += 1        

                                word2 += ch2
                            
                        elif scenario==3:
                            # сложные замены цепочек букв типа ТСЯ-ТЬСЯ
                            for (seq1,seq2) in pattern_replacement.items():
                                if word.find(seq1)!=-1:
                                    word2 = word.replace( seq1, seq2 )
                                    break    
                            
                        if word2!='' and not word2 in wordforms and len(word2)<=max_word_len:
                            padded_word = word2 + SENTINEL_CHAR * (max_word_len - len(word2))
                            left_word = padded_word
                            if INVERT:
                                left_word = left_word[::-1]
                            
                            left_words.append(left_word)
                            right_words.append(right_word)
                            misspellings_count += 1

                
            n_patterns = len(left_words)
            
            print( 'Total number of patterns=', n_patterns )
            print( 'misspellings_count=', misspellings_count )
            
            # теперь заполняем датасеты, строим модель и обучаем ее
            if model1_type=='char-rnn':
                print('Vectorization for char-rnn...')
                X_train = np.zeros((n_patterns, max_word_len, bits_per_char), dtype=np.bool)
                y_train = np.zeros((n_patterns, max_lemma_len, bits_per_char), dtype=np.bool)
                i_train = 0

                for i in range(n_patterns):

                    left_word = left_words[i]
                    right_word = right_words[i]

                    X_train[i_train] = ctable.encode(left_word, maxlen=max_word_len)
                    y_train[i_train] = ctable.encode(right_word, maxlen=max_lemma_len)
                    i_train += 1

                        
                print('Build model RNN(', str(max_word_len), '-->', str(max_lemma_len), ')...')
                model = Sequential()
                model.add( Masking( mask_value=0,input_shape=(max_word_len,bits_per_char) ) )
                model.add( RNN( HIDDEN_SIZE, input_shape=(max_word_len, bits_per_char), dropout_W=DROPOUT, dropout_U=DROPOUT ) )
                model.add( RepeatVector(right_len) )
                model.add( RNN( HIDDEN_SIZE, return_sequences=True ) )

                # For each of step of the output sequence, decide which character should be chosen
                model.add(TimeDistributedDense(bits_per_char))
                model.add(Activation('softmax'))

                open( CHAR_RNN_ARCH_FILENAME, 'w').write( model.to_json() )


                model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

                model_checkpoint = ModelCheckpoint( CHAR_RNN_WEIGHTS_FILENAME.replace( '#', p_o_s ), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
                early_stopping = EarlyStopping( monitor='val_loss', patience=5, verbose=1, mode='auto')

                vizualizer = VizualizeCallback_RNN1(p_o_s)

                history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=N_EPOCH0, validation_split=0.1, callbacks=[model_checkpoint,early_stopping,vizualizer] )
                
                hist_file = CHAR_RNN_HISTORY_FILENAME.replace( '#', p_o_s )
                print( u'Storing val_loss history to {0}'.format(hist_file) )
                np.savetxt( hist_file, history.history['val_loss'] )

            elif model1_type=='char-feedforward':
                print( 'Vectorization...' )
                
                input_size = max_word_len*bits_per_char
                output_size = max_lemma_len*bits_per_char
                
                X_train = np.zeros((n_patterns, input_size), dtype=np.bool)
                y_train = np.zeros((n_patterns, output_size), dtype=np.bool)
                i_train = 0

                for i in range(n_patterns):

                    left_word = left_words[i]
                    right_word = right_words[i]

                    X_train[i_train] = ctable.encode_feedforward(left_word, maxlen=max_word_len)
                    y_train[i_train] = ctable.encode_feedforward(right_word, maxlen=right_len)
                    i_train += 1

                print('Build model FEEDFORWARD(', str(input_size), '-->', str(output_size), ')...')
                model = Sequential()
                model.add( Dense( input_dim=max_word_len*bits_per_char, output_dim=HIDDEN_SIZE, activation=FF_ACTIVATION ) )
                model.add( Dense( output_dim=max_lemma_len*bits_per_char, activation=FF_ACTIVATION ) )

                open( CHAR_FEEDFORWARD_ARCH_FILENAME, 'w').write( model.to_json() )

                opt = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08)
                #opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile( loss='mean_squared_error', optimizer=opt )

                model_checkpoint = ModelCheckpoint( CHAR_FEEDFORWARD_WEIGHTS_FILENAME.replace( '#', p_o_s ), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
                early_stopping = EarlyStopping( monitor='val_loss', patience=5, verbose=1, mode='auto')

                vizualizer = VizualizeCallback_FEEDFORWARD1(p_o_s)

                history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=N_EPOCH0, validation_split=0.1, callbacks=[model_checkpoint,early_stopping,vizualizer] )

                hist_file = CHAR_FEEDFORWARD_HISTORY_FILENAME.replace( '#', p_o_s )
                print( u'Storing val_loss history to {0}'.format(hist_file) )
                np.savetxt( hist_file, history.history['val_loss'] )



















            
