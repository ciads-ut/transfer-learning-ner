# -*- coding: utf-8 -*-
""" This code was modified from nltk/tag/crf.py (Natural Language Toolkit:
Interface to the CRFSuite Tagger) to include POS tags.

Copyright (C) 2001-2017 NLTK Project
Author: Long Duong <longdt219@gmail.com>
URL: <http://nltk.org/>
For license information, see LICENSE.TXT (Apache v.2 license; can modify)

A module for POS tagging using CRFSuite

See http://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html#pycrfsuite.Trainer.select

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unicodedata
import re

from nltk.tag.api import TaggerI

try:
    import pycrfsuite
except ImportError:
    raise ImportError('Need the python-crfsuite in order to use CRFTagger')

class CRFTagger(TaggerI):
    """
    A module for POS tagging using CRFSuite https://pypi.python.org/pypi/python-crfsuite

    >>> from nltk.tag import CRFTagger
    >>> ct = CRFTagger()

    >>> train_data = [[('University','Noun'), ('is','Verb'), ('a','Det'), ('good','Adj'), ('place','Noun')],
    ... [('dog','Noun'),('eat','Verb'),('meat','Noun')]]

    >>> ct.train(train_data,'model.crf.tagger')
    >>> ct.tag_sents([['dog','is','good'], ['Cat','eat','meat']])
    [[('dog', 'Noun'), ('is', 'Verb'), ('good', 'Adj')], [('Cat', 'Noun'), ('eat', 'Verb'), ('meat', 'Noun')]]

    >>> gold_sentences = [[('dog','Noun'),('is','Verb'),('good','Adj')] , [('Cat','Noun'),('eat','Verb'), ('meat','Noun')]]
    >>> ct.evaluate(gold_sentences)
    1.0

    Setting learned model file
    >>> ct = CRFTagger()
    >>> ct.set_model_file('model.crf.tagger')
    >>> ct.evaluate(gold_sentences)
    1.0

    """


    def __init__(self,  feature_detector, verbose = False, training_opt = {}, algorithm = 'lbfgs'):
        """
        Initialize the CRFSuite tagger
        :param feature_detector: The function that extracts features for each token of a sentence. This function should take
        2 parameters: tokens and index which extract features at index position from tokens list. See the build in
        _get_features function for more detail.
        :param verbose: output the debugging messages during training.
        :type verbose: boolean
        :param training_opt: python-crfsuite training options
        :type training_opt : dictionary

        Set of possible training options (using LBFGS training algorithm).
         'feature.minfreq' : The minimum frequency of features.
         'feature.possible_states' : Force to generate possible state features.
         'feature.possible_transitions' : Force to generate possible transition features.
         'c1' : Coefficient for L1 regularization.
         'c2' : Coefficient for L2 regularization.
         'max_iterations' : The maximum number of iterations for L-BFGS optimization.
         'num_memories' : The number of limited memories for approximating the inverse hessian matrix.
         'epsilon' : Epsilon for testing the convergence of the objective.
         'period' : The duration of iterations to test the stopping criterion.
         'delta' : The threshold for the stopping criterion; an L-BFGS iteration stops when the
                    improvement of the log likelihood over the last ${period} iterations is no greater than this threshold.
         'linesearch' : The line search algorithm used in L-BFGS updates:
                           { 'MoreThuente': More and Thuente's method,
                              'Backtracking': Backtracking method with regular Wolfe condition,
                              'StrongBacktracking': Backtracking method with strong Wolfe condition
                           }
         'max_linesearch' :  The maximum number of trials for the line search algorithm.

        """

        self._model_file = ''
        self._tagger = pycrfsuite.Tagger()

        self._feature_func = self._featfun_to_listfun(feature_detector)

        self._verbose = verbose
        self._training_options = training_opt
        self._pattern = re.compile(r'\d')
        self._algorithm = algorithm

    def set_model_file(self, model_file):
        self._model_file = model_file
        self._tagger.open(self._model_file)


    def _featfun_to_listfun(self, feature_func):
        """ Convert the feature function (which returns a dict) to the form
        that is used by CRF, ie. a function that returs a list."""

        def listfun(tokens,index,history):
            features = feature_func(tokens, index, history)
            featurelist = []
            for k,v in features.items():
                featurelist.append(k+'_'+unicode(v))
            return featurelist

        return listfun


    def tag_sents(self, sents):
        '''
        Tag a list of sentences. NB before using this function, user should specify the mode_file either by
                       - Train a new model using ``train'' function
                       - Use the pre-trained model which is set via ``set_model_file'' function
        :params sentences : list of sentences needed to tag.
        :type sentences : list(list(str))
        :return : list of tagged sentences.
        :rtype : list (list (tuple(str,str)))
        '''
        history = []

        if self._model_file == '':
            raise Exception(' No model file is found !! Please use train or set_model_file function')

        # We need the list of sentences instead of the list generator for matching the input and output
        result = []
        for tokens in sents:
            features = [self._feature_func(tokens,i,history) for i in range(len(tokens))]
            labels = self._tagger.tag(features)

            if len(labels) != len(tokens):
                raise Exception(' Predicted Length Not Matched, Expect Errors !')

            tagged_sent = list(zip(tokens,labels))
            result.append(tagged_sent)

        return result


    def train(self, train_data, model_file):
        '''
        Train the CRF tagger using CRFSuite
        :params train_data : is the list of annotated sentences.
        :type train_data : list (list(tuple(str,str)))
        :params model_file : the model will be saved to this file.

        '''
        history = []
        trainer = pycrfsuite.Trainer(verbose=self._verbose, algorithm=self._algorithm) # before I used whatever the default was...
        print('Algorithm used by CRF: '+self._algorithm)
        trainer.set_params(self._training_options)

        for sent in train_data:
            tokens,labels = zip(*sent)
            features = [self._feature_func(tokens,i,history) for i in range(len(tokens))]
            trainer.append(features,labels)

        # Now train the model, the output should be model_file
        trainer.train(model_file)
        # Save the model file
        self.set_model_file(model_file)


    def tag(self, tokens):
        '''
        Tag a sentence using Python CRFSuite Tagger. NB before using this function, user should specify the mode_file either by
                       - Train a new model using ``train'' function
                       - Use the pre-trained model which is set via ``set_model_file'' function
        :params tokens : list of tokens needed to tag.
        :type tokens : list(str)
        :return : list of tagged tokens.
        :rtype : list (tuple(str,str))
        '''

        return self.tag_sents([tokens])[0]

