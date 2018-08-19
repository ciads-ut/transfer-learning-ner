""" This contains the function run_experiment, which can be used to run
the experiments in the COLING paper that do not involve neural networks.

To run the experiments involving the BiLSTM-CRFs, use train_bilstm_model.py
to train the model on the source dataset and save the model, and
load_pretrained.py to load the model and fine-tune on the target dataset.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import random
try:
    import ConfigParser
except:
    import configparser as ConfigParser
import pandas as pd

from nltk.data import path as nltk_data_path
nltk_data_location = os.getenv('NLTK_DATA_PATH')
if nltk_data_location is not None:
    nltk_data_path.append(nltk_data_location)
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

import utils
import transferlearning as tl
from stratified_split import writefile
from utils import _getlist


def run_experiment(transfer_exp_name):
    """ Run an experiment given in the experiments directory. Some
    configuration details (which seeds to use, which datasets to use,
    which transfer methods to use, which classifiers to use, number of
    sentences in the target training set, etc.) are given
    in the .cfg file within the directory transfer_exp_name.

    Parameters
    ----------

    transfer_exp_name : str
        The name of the set of transfer learning experiments.
        For example, CONLL03_to_GUM

    Remarks
    -------

    The settings in each Configuration file and the ones used in the COLING
    paper. However, when averaging the results, the run with seed 0 was ignored
    (a seed of 0 indicated the data was not shuffled).

    """
    TRAIN_DEV_SPLIT = 0.8 # 80%/20% train/dev split.
    SEP = ' '

    DIR = 'experiments/'+transfer_exp_name

    config_file = os.path.join(DIR, transfer_exp_name+'.cfg')

    config = ConfigParser.RawConfigParser(allow_no_value=True)
    config.read(config_file)

    classifier = config.get('algorithm', 'classifier')
    transfer_methods = _getlist(config, 'algorithm', 'transfer')

    src_names = _getlist(config, 'corpora', 'src_train')
    src_test_names = _getlist(config, 'corpora', 'src_test')

    tgt_train_names = _getlist(config, 'corpora', 'tgt_train')
    tgt_test_names = _getlist(config, 'corpora', 'tgt_test')

    src_label = ';'.join(src_names)
    tgt_label = ';'.join(tgt_train_names)

    src_train_shuffle_seed = config.get('corpora', 'src_train_shuffle_seed')
    tgt_train_shuffle_seeds = _getlist(config, 'corpora', 'tgt_train_shuffle_seeds', ints=True)

    src_train_length = config.get('split', 'src_train_length')
    tgt_test_length =  config.get('split', 'tgt_test_length')
    tgt_train_lengths = _getlist(config,'split', 'tgt_train_lengths', ints=True)

    # SOURCE DATA
    corpora = [list(utils.read_conll(i)) for i in src_names]
    src_train = [sent for corpus in corpora for sent in corpus]
    # This is only used to write the file:
    corpora = [list(utils.read_conll(i)) for i in src_test_names]
    src_test = [sent for corpus in corpora for sent in corpus]

    ## TARGET DATA
    corpora = [list(utils.read_conll(i)) for i in tgt_train_names]
    tgt_train = [sent for corpus in corpora for sent in corpus]
    corpora = [list(utils.read_conll(i)) for i in tgt_test_names]
    tgt_test = [sent for corpus in corpora for sent in corpus]

    # To shuffle or not to shuffle.
    if src_train_shuffle_seed is not None:
        src_train_shuffle_seed = int(src_train_shuffle_seed)
        src_train = shuffle_data(src_train, src_train_shuffle_seed)

    if tgt_train_shuffle_seeds is None:
        tgt_train_shuffle_seeds = [0] # 0 means no shuffle

    # Possibly use only first part of src training data.
    if src_train_length is not None:
        src_train = src_train[:int(src_train_length)]
    # Possibly use only first part of tgt testing data.
    if tgt_test_length is not None:
        tgt_test = tgt_test[:int(tgt_test_length)]
    if tgt_train_lengths is None:
        tgt_train_lengths = [len(tgt_train)]

    writefile(src_test, os.path.join(DIR,'src_data'),'test.txt', sep=SEP)
    src_train_, src_dev_ = split_corpus(src_train, TRAIN_DEV_SPLIT)
    writefile(src_train_, os.path.join(DIR, 'src_data'), 'train.txt', sep=SEP)
    writefile(src_dev_, os.path.join(DIR, 'src_data'), 'valid.txt', sep=SEP)

    num_reps = int(config.get('split', 'num_reps'))

    ent_excluded = _getlist(config, 'evaluation', 'excluded')
    if ent_excluded is not None:
        ent_excluded = set(ent_excluded)
    else:
        ent_excluded = set()

    src_train = utils.attach_domain(src_train, 'src')

    tgt_trainall = utils.attach_domain(tgt_train, 'tgt')
    tgt_test = utils.attach_domain(tgt_test, 'tgt')

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
    c = 0
    for tgt_tr_shuffle_seed in tgt_train_shuffle_seeds:
        tgt_trainall_shuff = shuffle_data(tgt_trainall, tgt_tr_shuffle_seed)
        for ttl in tgt_train_lengths:
            subdir = os.path.join(DIR, 'seed_'+str(tgt_tr_shuffle_seed), 'tgt_train_len_'+str(ttl)  )

            tgt_train, unused = split_corpus(tgt_trainall_shuff, ttl)
            tgt_train_, tgt_dev_ = split_corpus(tgt_train, TRAIN_DEV_SPLIT)

            writefile(tgt_train_, os.path.join(subdir, 'tgt_data'), 'train.txt', sep=SEP)
            writefile(tgt_dev_, os.path.join(subdir, 'tgt_data'), 'valid.txt', sep=SEP)
            writefile(tgt_test, os.path.join(subdir, 'tgt_data'), 'test.txt', sep=SEP)

            tgt_train_toks = sum([len(i) for i in tgt_train])

            for transfermethod in transfer_methods:
                subdir_transfer = os.path.join(subdir, 'method_'+transfermethod)
                #NOTE for now just run once, so don't make subdirectory for rep
                for rep in range(num_reps):
                    print 'Number of tgt training sentences: ', ttl
                    print 'Transfer method: ', transfermethod
                    print 'Run: ', rep

                    D = tl.DomainAdaptation(verbose=False)

                    if transfermethod[:4] == 'pred':
                        params = method_param_mappings(transfermethod)
                        D.train('pred', classifier, src_train, tgt_train, tgt_test, **params)
                    else:
                        D.train(transfermethod, classifier, src_train, tgt_train, tgt_test)
                    score = D.evaluate(tgt_test)
                    writefile(score.predicted, subdir_transfer, 'predicted.conll', sep=SEP)
                    score.write_report(os.path.join(subdir_transfer,'results.txt'), ent_excluded)

                    print "Trained and tested", transfermethod, " with", classifier
                    #print score.F1()
                    #############
                    res = score.microPRF1()
                    P, R, F1 = res
                    print 'microF1: ', F1
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
                results.to_pickle(os.path.join(DIR,'results.pkl'))

    return results


def readdata(src_corpus_name, tgt_corpus_name, shuffle=True, seed=34):
    """ Read in src and tgt data, and only shuffle the target data.

    """
    src_reader = utils.read_conll(src_corpus_name)
    tgt_reader = utils.read_conll(tgt_corpus_name)
    src_data, tgt_data = list(src_reader), list(tgt_reader)
    if shuffle: # Only shuffle tgt.
        tgt_data = shuffle_data(tgt_data, seed)

    return src_data, tgt_data


def shuffle_data(corpus, seed):
    """ Shuffle the data. corpus is list of lists (sentences) to be shuffled.

    Note: I will use the convention that seed 0 means no shuffle. If seed is
    None that also means no shuffle will occur.

    """
    if seed not in {None, 0}:
        random.seed(seed)
        shuffled = random.sample(corpus, len(corpus))
    else:
        shuffled = corpus

    return shuffled


def split_corpus(data, train_amount, test_length=None):
    """ Split the sentences into train and test sets. If test_length
    is None, then the split is:

    train = the first 'train_length' number of sentences.
    test = the rest.

    If test_length is given, the split is:

    train = the first 'train_length' number of sentences.
    test = the last 'test_length' number of sentences.

    NOTE: train_amount + test_length must be less than the total number of
    sentences.

    """
    if 0 <= train_amount <= 1:
        train_length = int(len(data) * train_amount)
    else:
        train_length = train_amount  # number of sentences in training set
    train = data[:train_length]

    if test_length is None:
        test = data[train_length :]
    else: # test_length is number of testing sentences
        if train_length + test_length > len(data):
            raise ValueError("The corpus is not long enough to have that much training & testing data.")
        test = data[len(data)-test_length : ]
    return train, test


def method_param_mappings(method):
    if method == 'pred':
        params = {'with_cca': False, 'no_prefix':False}

    if method == 'pred-no_prefix':
        params = {'with_cca': False, 'no_prefix':True}

    if method == 'predCCA':
        params = {'with_cca': True, 'no_prefix':False, 'exclude_O':False}

    if method == 'predCCA-no_prefix':
        params = {'with_cca': True, 'no_prefix':True, 'exclude_O':False}

    if method == 'predCCA-no_prefix-excludeO':
        params = {'with_cca': True, 'no_prefix':True, 'exclude_O':True}

    if method == 'predCCA-excludeO':
        params = {'with_cca': True, 'no_prefix':False, 'exclude_O':True}

    return params

