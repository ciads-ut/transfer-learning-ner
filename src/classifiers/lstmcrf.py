""" This is an implementation of a bi-LSTM-CRF using Keras.

Remark: to use this, make sure the keras backend is set to tensorflow.

"""
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import (LSTM, Embedding,
                         Dense, TimeDistributed,
                         Dropout, Bidirectional,
                         Lambda)
from keras_contrib.layers import CRF
from keras import optimizers
from keras.layers.merge import Concatenate
import keras.backend as K
from collections import Counter


def predict(model, test, tag2idx, word2idx, tags, max_len, batch_size=32):
    """ Use BiLSTM-CRF model to make new predictions.

    """
    # With case embeddings:
    X_te, y_te, X_ca = prepare_inputs_outputs(test, word2idx, tag2idx, max_len)
    ps = [model.predict( [ np.array([X_te[i]]), np.array([X_ca[i]]) ], batch_size=batch_size) for i in xrange(len(test))]
    ps = [np.argmax(x, axis=-1) for x in ps]

    A = [ [(test[jj][ii][0], tags[pred] ) for ii, pred in enumerate(ps[jj][0]) if X_te[jj][ii]!=0 ] for jj in xrange(len(test))]

    return A


#TODO change the name of this function, this is misleading.
def get_word2idx2(sentences):
    words = {}
    for sent in sentences:
        for tok, label in sent:
            words[tok[0].lower()] = True
    return words


def getCasing(word, case2Idx):
    """ Returns casing information for a given word.

    Remark: this function was adapted from:
    https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master/2016-11_Seminar/Session 1 - SENNA/

    """
    casing = 'other'
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
    digitFraction = numDigits / float(len(word))

    if word.isdigit(): # Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): # All lower case
        casing = 'allLower'
    elif word.isupper(): # All upper case
        casing = 'allUpper'
    elif word[0].isupper(): # Is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    return case2Idx[casing]


def get_tag2idx(sentences):
    tags = list(set([x[-1] for s in sentences for x in s]))
    #n_tags = len(tags)
    tag2idx = {t: i for i, t in enumerate(tags)}
    return tags, tag2idx


def get_maxlen(sentences):
    max_len = max([len(i) for i in sentences]) + 1
    return max_len


def prepare_inputs_outputs(sentences, word2idx, tag2idx, max_len):

    case2Idx = {'numeric': 7, 'allLower':1, 'allUpper':2, 'initialUpper':3,
                'other':4, 'mainly_numeric':5,
                'contains_digit': 6, 'PADDING_TOKEN':0}
    # NOTE the value for PADDING_TOKEN must be zero, we will mask it.

    wordset_in_corpus = set([x[0][0] for s in sentences for x in s])
    # Since using Glove embeddings, need to lower-case the words.
    X_tr = []
    X_ca = [] # casing info
    UNK_IND = word2idx['UNKNOWN_TOKEN']
    unknown_words = Counter()
    for s in sentences:
        S = []
        S_ca = []
        for x in s:
            S_ca.append(getCasing(x[0][0], case2Idx))
            word = x[0][0].lower()
            if word in word2idx:
                S.append(word2idx[word])
            else:
                unknown_words[word] += 1
                S.append(UNK_IND)
        X_tr.append(S)
        X_ca.append(S_ca)

    # NOTE in case the # of words in a sentence exceed max_len (rarely happens)
    # need to use truncating = "post", which removes all the words past max_len
    X_tr = pad_sequences(maxlen=max_len, sequences=X_tr, padding="post", value=0, truncating="post")

    X_ca = pad_sequences(maxlen=max_len, sequences=X_ca, padding="post", value=case2Idx["PADDING_TOKEN"], truncating="post")

    y_tr = [[tag2idx[x[1]] for x in s] for s in sentences]
    y_tr = pad_sequences(maxlen=max_len, sequences=y_tr, padding="post", value=tag2idx["O"], truncating="post")
    y_tr = [to_categorical(i, num_classes=len(tag2idx) ) for i in y_tr]

    #print 'Fraction of unknown words: ', float(len(unknown_words))/len(wordset_in_corpus)

    return X_tr, y_tr, X_ca #, unknown_words


def make_biLSTM_casing_CRF2(n_tags, n_words, max_len, pretrained_embeddings = None,
                            pre_dropout = 0.5,
                            LSTM_naive_dropout = None,
                            recurrent_dropout = None,
                            WORDVEC_DIM = 100,
                            LSTM_DIM = 100): # 100 is used in NeuroNER.
    """ Returns a BiLSTM-CRF keras model, with case embeddings.

    """
    case2Idx = {'numeric': 7, 'allLower':1, 'allUpper':2, 'initialUpper':3,
                'other':4, 'mainly_numeric':5,
                'contains_digit': 6, 'PADDING_TOKEN':0}

    caseEmbeddings =  np.identity(len(case2Idx))
    word_inputs = Input(shape=(max_len,), name ='input')

    word_embeddings = Embedding(input_dim=n_words,
                                output_dim=WORDVEC_DIM,
                                input_length=max_len,
                                mask_zero=True,
                                name = 'word_embedding'
                                )(word_inputs)

    case_inputs = Input(shape=(max_len,), name = 'input_casing')

    word_casings = Embedding(input_dim=caseEmbeddings.shape[0],
                             output_dim=caseEmbeddings.shape[1],
                             input_length=max_len,
                             mask_zero=True,
                             name = 'case_embedding',
                             trainable = False, # Of course these should not be trainable.
                             weights=[caseEmbeddings]
                             )(case_inputs)

    # Concatenate casing and word embeddings:
    word_case_embeddings =Concatenate(axis=-1)([word_embeddings, word_casings])

    if pre_dropout:
        print('Using pre dropout')
        word_case_embeddings = Dropout(pre_dropout)(word_case_embeddings)

    if not LSTM_naive_dropout:
        LSTM_naive_dropout = 0
    if not recurrent_dropout:
        recurrent_dropout = 0

    model = Bidirectional(LSTM(units=LSTM_DIM,
                               return_sequences=True,
                               recurrent_dropout=recurrent_dropout,
                               dropout=LSTM_naive_dropout,
                               name='LSTM'),
                               name='biLSTM'
                               )(word_case_embeddings)

    model = TimeDistributed(Dense(n_tags,
                                  activation=None,
                                  name='dense'),
                                  name='td'
                                  )(model)

    crf = CRF(n_tags)  # CRF layer # Do not give it a name.
    out = crf(model)   # output
    model = Model(inputs=[word_inputs, case_inputs], outputs=[out])

    if pretrained_embeddings is not None:
        ##model.layers[1].set_weights([pretrained_embeddings])
        model.get_layer(name='word_embedding').set_weights([pretrained_embeddings])
        # NOTE even though it's loading pretrained embeddings here, later on
        # in load_pretrained.py we load the weights of the old network which
        # replace these when reusing layers.
    return model, crf


def fit_model(model, X_tr,
              X_ca,
              y_tr,
              X_dev, y_dev, Xdev_ca,
              epochs,
              savename=None):

    early_stopping = EarlyStopping(monitor = 'val_loss',
                                   patience = 10)
    #print X_tr.shape
    #print X_ca.shape
    history = model.fit([X_tr, X_ca], # used to be X_tr
                        np.array(y_tr),
                        batch_size=32,
                        epochs=epochs,
                        callbacks = [early_stopping],
                        validation_data=([X_dev, Xdev_ca], np.array(y_dev) ),
                        verbose=1,
                        shuffle = False) # I already shuffled it.

    if savename:
        model.save_weights(savename)
    return history

