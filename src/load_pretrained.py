""" This can be used to load the pre-trained word embeddings and fine-tune
on another corpus.

Usage
-----

>>> make_reports(tgt_corpus, 'rmsprop_v3sgd', tlayers)

where tgt_corpus is the name of the target corpus on which to fine-tune. This
name should correspond to name of the directory in experiments. The options are:

    GUM
    TwitterRitter
    re3d
    NIST99
    MUC6
    MITRestaurant
    MITMovie
    i2b2-14
    i2b2-06
    CADEC

For tlayers use 'N': none, train from scratch
                'E': transfer the embedding layer only
                'EL': transfer both embedding layer and biLSTM layer
                      (the setting in the paper).

"""
#################################
# For reproducibility (though this is not entirely reproducible)
import numpy as np
import random
np.random.seed(42)
random.seed(12345)
import tensorflow as tf
tf.set_random_seed(1234)
#################################

import gzip
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras import optimizers
try:
    import cPickle as pkl
except:
    import _pickle as pkl  # Python 3
try:
    import ConfigParser
except:
    import configparser as ConfigParser

import utils
import classifiers.lstmcrf as lc
import evaluation
from utils import _getlist
from stratified_split import writefile

CONLLTAGSET = {'PER','LOC','ORG','MISC'}

MODEL_FILE = 'models_bilstmcrf/final_model_100_withcase3.h5'
EMBEDDINGS_FILE = 'models_bilstmcrf/embeddings_1.pkl.gz'

WVDIM = 100


def load_embeddings(embeddingsPath):
    """ Load all the things, except for the model."""

    with gzip.open(embeddingsPath, 'rb') as f:
        embeddings = pkl.load(f)
    t2i = embeddings['l2i']
    w2i = embeddings['w2i']
    we = embeddings['we']

    tags = ['']*len(t2i)
    for k,v in t2i.items():
        tags[v] = k

    return we, w2i, t2i, tags


def load_pretrained_model(tgt_corpus, seed, length, optimizer_name, tlayers):
    """ Load a pre-trained model and decide which layers should be transfered.

    For tlayers use 'N': none, train from scratch
                    'E': transfer embedding layer only
                    'EL': transfer both embedding layer and biLSTM layer

    """

    print('loading pretrained model')

    max_len = 253

    we, w2i, t2i, tags = load_embeddings(EMBEDDINGS_FILE)

    print('obtaining train data... ')
    trainfile = 'experiments/CONLL03_to_'+tgt_corpus+'/seed_'+str(seed)+'/tgt_train_len_'+str(length)+'/tgt_data/train.txt'
    testfile = 'experiments/CONLL03_to_'+tgt_corpus+'/seed_'+str(seed)+'/tgt_train_len_'+str(length)+'/tgt_data/test.txt'
    validfile = 'experiments/CONLL03_to_'+tgt_corpus+'/seed_'+str(seed)+'/tgt_train_len_'+str(length)+'/tgt_data/valid.txt'

    tgt_train = list( utils.read_NER_output(trainfile) )
    tgt_test = list( utils.read_NER_output(testfile) )
    tgt_dev = list( utils.read_NER_output(validfile) )

    tags_tgt, tag2idx_tgt = lc.get_tag2idx(tgt_train + tgt_test + tgt_dev)

    X_tr, y_tr, Xtr_ca = lc.prepare_inputs_outputs(tgt_train, w2i, tag2idx_tgt, max_len)
    X_te, y_te, Xte_ca = lc.prepare_inputs_outputs(tgt_test, w2i, tag2idx_tgt, max_len)
    X_dev, y_dev, Xdev_ca = lc.prepare_inputs_outputs(tgt_dev, w2i, tag2idx_tgt, max_len)

    model, crf = lc.make_biLSTM_casing_CRF2(len(tag2idx_tgt), len(w2i), max_len, we,
                                            WORDVEC_DIM=WVDIM)

    # This changes the name of the last layer so it doesn't match 'crf_1'.
    model.layers[-1].name = 'do_not_load_me'
    model.layers[-2].name = 'do_not_load_me'

    optimizer = optimizers.RMSprop(clipvalue=5.0)

    model.compile(optimizer = optimizer,
                  loss = crf.loss_function,
                  metrics = [crf.accuracy])

    if tlayers in {'E', 'EL'}:
        if tlayers == 'E':
            model.get_layer(name='biLSTM').name = 'do_not_load_me'
            print("Only transfer the first (word embedding) layer.")
        if tlayers == 'EL':
            print("Transfer both the word embedding and the biLSTM layer.")
        model.load_weights(MODEL_FILE, by_name = True)
    elif tlayers == 'N':
        print("Not reusing any layers. Training from scratch.")
    else:
        raise ValueError("!!")

    return model, X_tr, Xtr_ca, y_tr, X_dev, y_dev, Xdev_ca, tgt_test, tag2idx_tgt, w2i, tags_tgt, max_len


def fine_tune(model, X_tr, Xtr_ca, y_tr, X_dev, y_dev, Xdev_ca, tgt_test, tag2idx_tgt, w2i, tags_tgt, max_len, epochs=100):
    """ Fine-tune and predict.

    """
    np.random.seed(42)
    random.seed(12345)
    tf.set_random_seed(1234)

    history = lc.fit_model(model, X_tr, Xtr_ca, y_tr, X_dev, y_dev, Xdev_ca, epochs)
    print('Evaluating...')
    pred = lc.predict(model, tgt_test, tag2idx_tgt, w2i, tags_tgt, max_len)
    score = evaluation.Evaluator(pred, tgt_test, CONLLTAGSET)
    return history, score


def make_report(tgt_corpus, seed, length, optimizer_name, tlayers):
    model, X_tr, Xtr_ca, y_tr, X_dev, y_dev, Xdev_ca, tgt_test, tag2idx_tgt, w2i, tags_tgt, max_len  = load_pretrained_model(tgt_corpus, seed, length, optimizer_name, tlayers)
    history, score = fine_tune(model, X_tr, Xtr_ca, y_tr, X_dev, y_dev, Xdev_ca, tgt_test, tag2idx_tgt, w2i, tags_tgt, max_len, epochs=100)

    config_file = 'experiments/CONLL03_to_'+tgt_corpus+'/CONLL03_to_'+tgt_corpus+'.cfg'

    config = ConfigParser.RawConfigParser(allow_no_value=True)
    config.read(config_file)

    ent_excluded = _getlist(config, 'evaluation', 'excluded')
    if ent_excluded is not None:
        ent_excluded = set(ent_excluded)
    else:
        ent_excluded = set()

    write_dir = 'experiments/CONLL03_to_'+tgt_corpus+'/seed_'+str(seed)+'/tgt_train_len_'+str(length)+'/method_pretrain' + tlayers + '_' + optimizer_name +'/'
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    score.write_report(write_dir+'results.txt', ent_excluded)
    writefile(score.predicted, write_dir, 'predicted.conll')

    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'], 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    fig.savefig(write_dir+'epoch_history.png')

    return history, score, ent_excluded


def make_reports(tgt_corpus, optimizer_name, tlayers):
    """ For optimizer use lowercase, eg. adagrad, adam, rmsprop

    For optimizer_name, use 'rmsprop_v3sgd'

    For tlayers use 'N': none, train from scratch
                    'E': embedding layer only
                    'EL': embedding layer and biLSTM layer

    """

    transfermethod = 'finetune' + tlayers +'_' + optimizer_name
    classifier='biLSTM_CRF'

    config_file = 'experiments/CONLL03_to_'+tgt_corpus+'/CONLL03_to_'+tgt_corpus+'.cfg'
    config = ConfigParser.RawConfigParser(allow_no_value=True)
    config.read(config_file)

    # tgt train lengths
    lengths = _getlist(config,'split', 'tgt_train_lengths', ints=True)
    seeds = [1, 2, 3, 4, 5]  #NOTE don't use seed 0 when averaging.

    src_names = _getlist(config, 'corpora', 'src_train')
    src_label = ';'.join(src_names)

    tgt_train_names = _getlist(config, 'corpora', 'tgt_train')
    tgt_label = ';'.join(tgt_train_names)
    rep=0

    results = pd.DataFrame(columns=['src',
                                    'tgt',
                                    'tgt_train-sents',
                                    'tgt_train-toks',
                                    'i',
                                    'tgt_seed',
                                    'classifier',
                                    'transfer',
                                    'acc',
                                    'P',
                                    'R',
                                    'F1',
                                    'macroP',
                                    'macroR',
                                    'macroF1',
                                    'micro-novel-P',
                                    'micro-novel-R',
                                    'micro-novel-F1',
                                    'macro-novel-P',
                                    'macro-novel-R',
                                    'macro-novel-F1'
                                    ])
    c=0
    for seed in seeds:
        for ttl in lengths:
            print('Number of tgt training sentences: '+str(ttl))

            trainfile = 'experiments/CONLL03_to_'+tgt_corpus+'/seed_'+str(seed)+'/tgt_train_len_'+str(ttl)+'/tgt_data/train.txt'
            tgt_train = list( utils.read_NER_output(trainfile) )
            tgt_train_toks = sum([len(i) for i in tgt_train])
            # NOTE the number of words will be different than for the other
            # experiments since this is only words in train, not dev, whereas
            # in the non-neural network experiments it was for both.
            tgt_tr_shuffle_seed = seed

            history, score, ent_excluded = make_report(tgt_corpus, seed, ttl,
                                                       optimizer_name,
                                                       tlayers)
            print('excluded: '+str(ent_excluded))

            res = score.microPRF1()
            P, R, F1 = res
            #print 'microF1: ', F1
            acc = score.accuracy()
            macroP, macroR, macroF1 = score.macroPRF1()
            #TODO what if there are no novel types???
            macro_novP, macro_novR, macro_novF1 = score.macroPRF1_noveltypes(ent_excluded)
            micro_novP, micro_novR, micro_novF1 = score.microPRF1_noveltypes(ent_excluded)
            results.loc[c] = [src_label, tgt_label, ttl, tgt_train_toks,
                              rep, tgt_tr_shuffle_seed, classifier,
                              transfermethod,
                              acc,
                              P,
                              R,
                              F1,
                              macroP,
                              macroR,
                              macroF1,
                              micro_novP,
                              micro_novR,
                              micro_novF1,
                              macro_novP,
                              macro_novR,
                              macro_novF1,
                              ]
            c += 1

            results.to_pickle('experiments/CONLL03_to_'+tgt_corpus+ '/results_pretrain' + tlayers + '_' + optimizer_name +'.pkl')

    return results

