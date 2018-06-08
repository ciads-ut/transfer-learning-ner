""" This contains functions to train and test a BiLSTM-CRF on the source corpus,
and to save both the trimmed word embeddings used and the model weights,
which can be used to initialize another neural net for fine tuning.

Usage
-----

>>> max_len, we, w2i, words = get_embeddings()
>>> # Train the biLSTM-CRF, and save to disk:
>>> history, score = fit_and_test_model(max_len, we, w2i, words)

"""

#################################
# NOTE: this was an attempt at making the neural network runs as reproducible
# as possible.
import numpy as np
import random
np.random.seed(42)
random.seed(12345)
import tensorflow as tf
tf.set_random_seed(1234)
#################################

from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras import optimizers
import gzip
import utils
try:
    import cPickle as pkl
except:
    import _pickle as pkl  # Python 3

import embedding_utils
import evaluation
import classifiers.lstmcrf as lc

WVDIM = '100'

def get_embeddings():
    """ Obtain and trim the word embeddings in both the source and target
    datasets.

    This will try to use the vocabularies of: 'GUM', 're3d', 'BBN', 'i2b2-14',
    'BBN','i2b2-14', 'i2b2-06', 'CADEC', 'TwitterRitter', 'MITRestaurantCorpus',
    'MITMovieCorpus-trivia10k13', 'MUC6', 'NIST_IEER99', 'GMB1', as well as
    for CONLL 2003.

    If a dataset is not found, it is skipped over.

    """
    embeddingsPath = 'word_embeddings/glove.6B.'+WVDIM+'d.txt.gz'

    print("Getting vocab from various datasets...")

    dnames = ['GUM', 're3d', 'BBN','i2b2-14',
              'i2b2-06', 'CADEC', 'TwitterRitter', 'MITRestaurantCorpus',
              'MITMovieCorpus-trivia10k13', 'MUC6', 'NIST_IEER99', 'GMB1']

    try:
        conll03 = list(utils.read_conll('CONLL03'))
    except:
        raise ValueError("Could not find CONLL 2003 dataset.")

    aggregation = []
    for dname in dnames:
        try:
            dataset = list(utils.read_conll(dname))
            aggregation.extend(dataset)
        except:
            print(dname + " could not be found.")

    aggregation = conll03 + aggregation

    words = lc.get_word2idx2(aggregation)
    max_len = lc.get_maxlen(aggregation)

    # NOTE: max_len was 253 for our experiments.

    print("Getting word embeddings...")
    we, w2i = embedding_utils.get_word_embeddings(embeddingsPath, words)

    return max_len, we, w2i, words


def fit_and_test_model(max_len, we, w2i, words):
    """ Fit and test a BiLSTM-CRF on the CONLL 2003 corpus. Return both the
    training history and the score (evaluated on the source testing file).

    """
    # NOTE Using the custom train/dev/test split.
    print("Obtaining train data...")
    trainfile = 'experiments/CONLL03_to_GUM/src_data/train.txt'
    testfile = 'experiments/CONLL03_to_GUM/src_data/test.txt'
    validfile = 'experiments/CONLL03_to_GUM/src_data/valid.txt'
    src_train = list(utils.read_NER_output(trainfile))
    src_test = list(utils.read_NER_output(testfile))
    src_dev = list(utils.read_NER_output(validfile))

    tags_src, tag2idx_src = lc.get_tag2idx(src_train + src_test + src_dev)

    X_tr, y_tr, Xtr_ca = lc.prepare_inputs_outputs(src_train, w2i, tag2idx_src, max_len)
    X_te, y_te, Xte_ca = lc.prepare_inputs_outputs(src_test, w2i, tag2idx_src, max_len)
    X_dev, y_dev, Xdev_ca = lc.prepare_inputs_outputs(src_dev, w2i, tag2idx_src, max_len)

    print 'Saving the word embeddings for use later.'

    embeddings = {'we': we,
                  'w2i': w2i,
                  'l2i': tag2idx_src}

    embedding_utils.pkl_save('models_bilstmcrf/embeddings_1.pkl.gz',
                        [embeddings],
                        "Embeddings")

    model, crf = lc.make_biLSTM_casing_CRF2(len(tag2idx_src),
                                            len(w2i),
                                            max_len,
                                            we,
                                            WORDVEC_DIM=int(WVDIM))

    optimizer = optimizers.SGD(lr=0.005, clipvalue=5.0)

    model.compile(optimizer = optimizer,
                  loss = crf.loss_function,
                  metrics = [crf.accuracy])

    history = lc.fit_model(model, X_tr,
                           Xtr_ca,
                           y_tr,
                           X_dev, y_dev, Xdev_ca,
                           int(WVDIM),
                           "models_bilstmcrf/final_model_100_withcase3.h5")

    print("Finished fitting the model. Going to predict now...")
    pred = lc.predict(model, src_test, tag2idx_src, w2i, tags_src, max_len)
    score = evaluation.Evaluator(pred, src_test, set())

    return history, score


