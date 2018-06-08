# -*- coding: utf-8 -*-
"""
Module for transfer learning for named entity recognition (NER).

Please note that this is only for the fully supervised case, where both the
source and target domains have labeled data.

For neural network approaches, see train_bilstm_model.py and load_pretrained.py

Transfer learning methods implemented here:
    - src
    - tgt
    - all
    - augment: Daume's EasyAdapt
    - PRED
    - PRED-CCA

Usage
-----

# quickstart will load some data for demo purposes.
>>> import quickstart as q

# To train a model on some source corpus and evaluate it on a target corpus:
>>> import transferlearning as tl
>>> D = tl.DomainAdaptation()
>>> D.train('augment', 'CRF', q.src, q.tgt_train, q.tgt_test)
>>> score = D.evaluate(q.tgt_test)
>>> score.F1()

# To load a pre-trained model and evaluate it on a corpus:
>>> import transferlearning as tl
>>> D = tl.DomainAdaptation()
>>> D.load_pretrained_model(modelname='pretrained-StanfordNER')
>>> score = D.evaluate(q.src)
>>> score.F1()

"""
import numpy as np
import os, sys
from tqdm import tqdm
from nltk.tag.sequential import ClassifierBasedTagger
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.chunk.util import ChunkScore
# Some classifiers in nltk for sequential classification:
from nltk.classify import megam, NaiveBayesClassifier, MaxentClassifier
from nltk.tag import StanfordNERTagger, SennaNERTagger
from nltk import ne_chunk as nltk_ner

from features import (ZhangJohnson,
                      wordembeddings_as_features,
                      combine_several_featfunctions)

from embedding_utils import (get_word_embeddings,
                             get_unique_words,
                             _get_correct_open)
from label_mismatch import cca
import BIO_utils
import sentence_utils
from evaluation import Evaluator
from classifiers.averaged_perceptron import PerceptronNER
from classifiers.crf import CRFTagger

# NLTK DATASETS & TOOLS
##########################
from nltk.data import path as nltk_data_path
nltk_data_location = os.getenv('NLTK_DATA_PATH')
if nltk_data_location is not None:
    nltk_data_path.append(nltk_data_location)

# EXTERNAL LIBRARIES
##########################
# NOTE Set this directory to wherever megam, MITIE, Stanford NER and SENNA are
# located.
EXTLIB_DIR = '/Data/nlp/utilities/'

#TODO this might need a try/except as well:
MEGAM_DIR = EXTLIB_DIR+'megam_0.92/'
try:
    megam.config_megam(MEGAM_DIR+'megam.opt')
except:
    print("megam is not installed or not configured correctly.")

MITIE_DIR = EXTLIB_DIR+'MITIE/'
MITIE_LIB_DIR = MITIE_DIR + 'mitielib/'
sys.path.append(MITIE_LIB_DIR)

try:
    import mitie
except:
    print("To use the pre-trained MITIE model, you will need to install the "+
           "MITIE Python wrapper.")
try:
    stanford_ner_path = os.environ.get('CLASSPATH')
except:
    print('You will need to set the CLASSPATH for Stanford NER if using their \
           pre-trained models.')

# PRE-TRAINED MODELS
##########################
STANFORD_MODEL_DIR = EXTLIB_DIR + 'StanfordNER/stanford-ner-2017-06-09/classifiers/'
SENNA_DIR = EXTLIB_DIR + 'senna/'
MITIE_MODEL_DIR = MITIE_DIR + 'MITIE-models/english/ner_model.dat'

# WORD EMBEDDINGS
##########################
embeddingsPath = 'word_embeddings/glove.6B.100d.txt.gz'

class DomainAdaptation():
    """ This will make it easy to compare different NER classifiers, domain,
    adaptation techniques and features. One can train the model on a corpus
    or use a pre-trained model, and test the method on a labeled dataset.

    """
    def __init__(self, verbose=False):
        self._verbose = verbose
        self.pretrained_model = 'None' # This is changed to something else if using a pretrained model.
        self.model = {}


    def _set_parameter(self,paramname, classifier_name, defaultvalue):
        """ Raise ValueError if the wrong parameter name (paramname) is given and
        the dictionary self.parameters in not empty, and
        save the new parameters into self.parameters. If no parameters are given
        the defaultvalue is used.

        This is used by train method of DomainAdaptation to set default parameters.

        Parameters
        ----------
        paramname : str, name of parameter to set
        defaultvalue : the default value
        """
        if not self.parameters.has_key(paramname) and len(self.parameters.keys())!=0:
            self.parameters={}
            raise ValueError('Optional argument for '+ classifier_name + ' must be '+ paramname)
        else:
            param = self.parameters.get(paramname, defaultvalue)
            self.parameters[paramname] = param


    def get_featurefunc(self, features_name, transfer_method, src_train,
                        tgt_train, tgt_test):
        if features_name == 'ZhangJohnson':
            features = ZhangJohnson
        elif features_name == 'word_embedding':
            if transfer_method =='src': allsentences = src_train + tgt_test
            if transfer_method =='tgt': allsentences = tgt_train + tgt_test
            if transfer_method =='_pred': allsentences = tgt_train + tgt_test
            if transfer_method in ['all','augment','pred']:
                allsentences = src_train + tgt_train + tgt_test
            allwords = get_unique_words(allsentences)
            print 'Obtaining word embedding information.'
            wordEmbeddings, word2Idx = get_word_embeddings(embeddingsPath, allwords)
            features = wordembeddings_as_features(wordEmbeddings, word2Idx)
            print 'Done obtaining word embeddings to use as features.'
        else:
            raise ValueError("features name is incorrect.")
        return features


    def train(self, transfer_method, classifier_name,
              src_train, tgt_train, tgt_test,
              features_name='ZhangJohnson', **kwargs):
        """ Train the model with a given classifier and with a given domain
        adaptation method (preprocessing or post-processing).

        Parameters
        ----------
        transfer_method : str
            The name of the transfer method to use. They can be:
                * src: Train the model with source training data
                * tgt: Train the model with target training data
                * all: Train the model with both source and target data
                * augment:
                    Train the model both source and target data, but
                    enlarge the feature space, using Daume's easyadapt
                    method[1], so if a token i is in the source,
                    use feature (x_i, x_i, 0) instead of x_i for each feature;
                    if the token in the target data use feature (x_i, 0, x_i).
                    The first entry of the tuple stands for 'general' features,
                    the second is 'source only', and the third is 'target only'.
                 * pred: The 'PRED' method, described in Daume (#TODO put other
                    references in here).
                    Unlike the other methods, the train function both trains
                    and tests, saves the result. Calling 'test' merely prints
                    the score.
                    This permits another optional keyword argument, 'no_prefix':
                    if True, it removes the 'I-' or 'B-' from the PRED feature.


        classifier_name : str
            The name of the classifier to use. Roughly in order of performance:
                * CRF: the CRFTagger from nltk, which calls external CRFSuite.
				  Optional keyword parameter: 'algorithm', which can be either
				  'l2sgd' or 'lbfgs'. If not given, 'l2sgd' is used.
                * averaged_perceptron : the averaged perceptron from nltk
                * megam: nltk's binding from of Daume's external megam program
                * IIS: Improved Iterative Scaling, via nltk
                * GIS: Generalized Iterative Scaling, via nltk
                * naivebayes: Naive Bayes from nltk.

        features_name : str or list
            Which features to use. Can be:
                * 'ZhangJohnson': The features used in Zhang and Johnson (2003).
                * 'word_embedding': Word embedding only.
                * a list containing any combination of the above options.

        src_train, tgt_train, tgt_test : lists
            Each of these is a list of lists, with entries of the form:
                    ( (word, pos, domain), entity )
            For now tgt_test is needed as an argument in order to get the full
            vocabulary for word embeddings.

        **kwargs:
            if classifier_name is 'averaged_perceptron':
                'num_iterations', default: 5 (same as nltk's default)
            if classifier_name is 'megam', 'IIS', or 'GIS':
                'gauss_prior_sigma', default: 1.0 (same as nltk's default)

        References
        ----------
        [1] Daumé III, Hal. "Frustratingly easy domain adaptation." arXiv
            preprint arXiv:0907.1815 (2009).

        [2] L-BFGS: http://aria42.com/blog/2014/12/understanding-lbfgs
                    http://www.umiacs.umd.edu/~hal/docs/daume04cg-bfgs.pdf

        Remarks
        -------
        On speed and memory:
            * megam is slow and memory intensive, though using the optimized
              megam (megam opt) does help.
            * IIS and GIS are pure python and slower than megam.

        """

    #######################################################################
    ## Store model information
    #######################################################################

        self.model['entity_types_src']= sentence_utils.get_tagset(src_train, with_prefix = False)
        self.model['entity_types_tgt']= sentence_utils.get_tagset(tgt_test+tgt_train, with_prefix = False)

        #self.classifier = classifier_name #unused.
        self.transfer_method = transfer_method
        self.parameters = kwargs
        #print '... called train. These are the parameters: ', self.parameters
        #self.featurelist = features.keys()
        if self._verbose:
            print "Transfer Learning: ",transfer_method, "  Classifier: ", classifier_name
        print 'kwargs', kwargs
        print 'exclude_O', self.parameters.get('exclude_O')
    #######################################################################
    ## Determine which features to use
    #######################################################################
    #TODO make an option so can choose whether to augment the word-embeddings with the other features or not...
        if isinstance(features_name,str):
            features = self.get_featurefunc(features_name, transfer_method, src_train, tgt_train, tgt_test)

        if isinstance(features_name,list):
            featureslist = []
            for featname in features_name:
                f = self.get_featurefunc(featname, transfer_method, src_train, tgt_train, tgt_test)
                featureslist.append(f)
            print 'Combining features...'
            features = combine_several_featfunctions(featureslist)

    #######################################################################
    ## Transfer Learning Options (specify training data & preprocessing)
    #######################################################################

        if transfer_method in ['src','tgt','all']:
            features_used = features
            if transfer_method=='src': train_data = src_train
            if transfer_method=='tgt': train_data = tgt_train
            if transfer_method=='all': train_data = src_train + tgt_train #self.all_train

        elif transfer_method == 'augment':
            train_data = src_train + tgt_train #self.all_train
            def augment_features(tokens, index, history):
                word, pos, domain = tokens[index]
                fts = features(tokens, index, history)
                for key in fts.keys():
                    if domain == 'src':
                        fts[domain + '-' + key] = fts[key]
                        fts['tgt' + '-' + key] = 0
                    else:
                        fts[domain + '-' + key] = fts[key]
                        fts['src' + '-' + key] = 0
                return fts
            features_used = augment_features

        elif transfer_method == '_pred': # this is not to be called directly;
        # It is used by pred, to train the second classifier.
            train_data = tgt_train

            no_prefix = self.parameters.get('no_prefix')
            with_cca = self.parameters.get('with_cca')
            kdim = self.parameters.get('kdim')
            exclude_O = self.parameters.get('exclude_O')

            if with_cca:
                label2vec = cca(src_train + tgt_train,
                                no_prefix = no_prefix,
                                k = kdim,
                                exclude_O = exclude_O)

            def pred_features(tokens, index, history):
                PRED = tokens[index][3]
                fts = features(tokens, index, history)
                if with_cca:
                    for i in range(kdim):
                        fts['PRED-cca-'+str(i)] = label2vec[PRED][i]
                else:
                    fts['PRED'] = PRED

                return fts

            def pred_features_noprefix(tokens, index, history):
                PRED = tokens[index][3]
                fts = features(tokens, index, history)
                # remove prefix 'I-' or 'B-':
                if PRED!='O':
                    PRED = PRED[2:]

                if with_cca:
                    for i in range(kdim):
                        fts['PRED-cca-'+str(i)] = label2vec[PRED][i]
                else:
                    fts['PRED'] = PRED

                return fts

            if no_prefix:
                features_used = pred_features_noprefix
            else:
                features_used = pred_features # default

        elif transfer_method == 'pred':
            no_prefix = self.parameters.get('no_prefix')
            with_cca = self.parameters.get('with_cca')
            kdim = self.parameters.get('kdim')
            if kdim is None:
                kdim = 5
            exclude_O = self.parameters.get('exclude_O')

            # TODO test this (i.e., using two different classifiers)
            if isinstance(classifier_name, list): # names of the two classifiers, in order
                classifier_name1 = classifier_name[0]
                classifier_name2 = classifier_name[1]
            else:
                classifier_name1 = classifier_name2 = classifier_name

            print('Training first classifier.')
            self.train('src', classifier_name1, src_train, tgt_train, tgt_test,
                       features_name = features_name)

            # FIRST: Use classifier on both the tgt_test and tgt_train
            print('Tagging tgt test data.')
            test_input_sentences = [zip(*t)[0] for t in tgt_test]
            test_predsents = self.NER.tag_sents(test_input_sentences)
            # flatten them:
            test_augmented = [ [tuple( list(f)+[zip(*p)[1][i]] ) for i,f in enumerate(zip(*p)[0])] for p in test_predsents]
            tgt_test =  [zip(x, [iob for (x,iob) in tgt_test[i]]) for i,x in enumerate(test_augmented)]
            # This is a list of lists of the form ((word, pos, dom, pred), iob)

            print('Tagging tgt train data.')
            train_input_sentences = [zip(*t)[0] for t in tgt_train]
            train_predsents = self.NER.tag_sents(train_input_sentences)
            train_augmented = [ [tuple( list(f)+[zip(*p)[1][i]] ) for i,f in enumerate(zip(*p)[0])] for p in train_predsents]
            tgt_train = [zip(x, [iob for (x,iob) in tgt_train[i]]) for i,x in enumerate(train_augmented)]

            # SECOND: train another classifier on the tgt_train data, with
            # the appended features from the first classifier.
            print('Training second classifier.\n')
            self.train('_pred', classifier_name2, src_train, tgt_train, tgt_test,
                features_name=features_name,
                kdim = kdim,
                no_prefix = no_prefix,
                with_cca = with_cca,
                exclude_O = exclude_O
                )
            #features_used = features # the features.py takes care of it
            classifier_name = 'none' # to prevent from continuing a second time.

            #self.predscore = self.test(tgt_test)
            self.predscore = self.evaluate(tgt_test)
            ##print self.predscore
            self.transfer_method = 'pred' # because the recursion will have changed it.

        else:
            pass

    #######################################################################
    ## Classifier Options: specifies which classifier to use and train
    #######################################################################
        # With 'megam, 'IIS', 'GIS' and 'naivebayes', will use
        # ClassifierBasedTagger to train the model.
        if classifier_name in ['megam','IIS','GIS','naivebayes']:
            if classifier_name == 'naivebayes':
                print "Training the model now..."
                classifier = NaiveBayesClassifier.train
                # NOTE Naive bayes works poorly with augment (due to the
                # breaking down of the independence assumption). This is
                # described in:
                #      Sutton and McCallum, An Introduction to Conditional
                #      Random Fields, p.16.

            if classifier_name in ['megam','IIS','GIS']:
                print "Training the model now..."
                if classifier_name in ['IIS','GIS']:
                    print("megam is recommended instead of IIS or GIS.")
                # NOTE: Though GIS and IIS cases also take gaussian_prior_sigma,
                #       they don't use it.  It only applies to megam.
                self._set_parameter('gauss_prior_sigma', classifier_name, 1.0)
                gauss_prior_sigma = self.parameters['gauss_prior_sigma']
                classifier = lambda traindata: MaxentClassifier.train(
                        traindata,
                        algorithm = classifier_name,
                        gaussian_prior_sigma = gauss_prior_sigma,
                        trace = 3*self._verbose)

            self.NER = ClassifierBasedTagger(
                train = train_data,
                feature_detector = features_used,
                classifier_builder = classifier,
                verbose = self._verbose,
                )

        if classifier_name == 'averaged_perceptron':
            print "Training the model now..."
            self._set_parameter('num_iterations', classifier_name, 5)
            num_iter = self.parameters['num_iterations']

            self.NER = PerceptronNER(
                feature_detector = features_used,
                verbose = self._verbose)
            self.NER.train(train_data, num_iterations = num_iter)

        if classifier_name == 'CRF':
            crfalgorithm = self.parameters.get('algorithm')
            if crfalgorithm is None:
                crfalgorithm = 'lbfgs' #'l2sgd'
                self.parameters['algorithm'] = crfalgorithm
            else:
                if crfalgorithm not in {'l2sgd', 'lbfgs'}:
                    raise ValueError("algorithm must be l2sgd' or 'lbfgs'.")

            print "Training the model now..."
            self.NER = CRFTagger(
                    feature_detector = features_used,
                    verbose = self._verbose, # more training options possible.
                    algorithm = crfalgorithm #'lbfgs' #'l2sgd' # lbfgs
                    )
            self.NER.train(train_data, 'model.crf.tagger')

        if classifier_name not in {'CRF', 'averaged_perceptron', 'megam',
                                   'IIS', 'GIS', 'naivebayes', 'none'}:
            raise ValueError("Wrong classifier name.")


    def load_pretrained_model(self, modelname='pretrained-StanfordNER',
                                    numclass=3):
        """ Loads a pre-trained model.

        Parameters
        ----------
        modelname : str
            The name of the pre-trained model to use. The options are:
                * 'pretrained-StanfordNER': Used a CRF and word embeddings.
                    See: https://nlp.stanford.edu/software/CRF-NER.shtml
                * 'pretrained-MITIE': Used Structural SVMs and word embeddings.
                    Uses Dhillon et al's "eigenwords" word embeddings.
                    See: https://github.com/mit-nlp/MITIE
                * 'pretrained-SENNA': Used multilayer perceptrons and the
                    50-dimensional CW (2008) word embeddings.
                    See: http://ml.nec-labs.com/senna/
                * 'pretrained-spacy': Used BILOU scheme; the algorithm is "a
                    pastiche of well-known methods...a greedy transition-based
                    parser guided by a linear model whose weights are learned
                    using the averaged perceptron loss, via the dynamic oracle
                    imitation strategy". See:
                    https://spacy.io/docs/usage/entity-recognition.
                    Using pre-trained model 'en_core_web_sm' here.
                    NOTE: could try 'en_depent_web_md' instead.
        numclass : int
            The number of classes for the pre-trained classifier; this is
            relevant only when modelname is 'pretrained-StanfordNER'.

        """
        self.pretrained_model = modelname
        self.transfer_method = 'none'
        if modelname == 'pretrained-StanfordNER':
            if numclass == 3:
                self.NER = StanfordNERTagger(STANFORD_MODEL_DIR+'english.all.3class.distsim.crf.ser.gz')#,
                        #STANFORD_CLASSPATH)
                self.model['entity_types'] = ['LOC',
                                              'ORG',
                                              'PER']
                self.model['training_corpus'] = ['CONLL03 eng.train',
                                                 'MUC6 train',
                                                 'MUC7 train',
                                                 'ACE2002',
                                                 'in-house data']
            elif numclass == 4:
                self.NER = StanfordNERTagger(STANFORD_MODEL_DIR+'english.conll.4class.distsim.crf.ser.gz')#,
                        ##STANFORD_CLASSPATH)
                self.model['entity_types'] = ['LOC','PER','ORG','MISC']
                self.model['training_corpus'] = ['CONLL03 eng.train']
            elif numclass == 7:
                self.NER = StanfordNERTagger(STANFORD_MODEL_DIR+'english.muc.7class.distsim.crf.ser.gz')#,
                        ##STANFORD_CLASSPATH)
                self.model['entity_types']=['LOC',
                                            'ORG',
                                            'PER',
                                            'MISC',
                                            'MON', # MONEY
                                            'PCT', # PERCENT
                                            'DAT', # DATE
                                            'TIM'] # TIME
                self.model['training_corpus'] = ['MUC6 train','MUC7 train']
            else:
                raise ValueError('When using StanfordNER, numclass must be 3, 4 or 7.')

        elif modelname == 'pretrained-MITIE':
            self.NER = mitie.named_entity_extractor(MITIE_MODEL_DIR)
            self.model['entity_types'] = ['PER', 'LOC', 'ORG', 'MISC']
            self.model['training_corpus'] = ['?']

        elif modelname == 'pretrained-SENNA':
            self.NER = SennaNERTagger(SENNA_DIR)
            self.model['entity_types'] = ['PER', 'LOC', 'ORG', 'MISC']
            self.model['training_corpus'] = ["?"]

        elif modelname == 'pretrained-spacy':
            self.NER = None
            self.model['entity_types']= ['PER', # PERSON
                                         'NOR', # NORP
                                         'FAC', # FACILITY
                                         'ORG', # ORGANIZATION
                                         'GPE', # GEO-POLITICAL
                                         'LOC', # LOCATION
                                         'PRO', # PRODUCT
                                         'EVE', # EVENT
                                         'WOR', # WORK OF ART
                                         'LAN', # LANGUAGE
                                         'DAT', # DATE
                                         'TIM', # TIME
                                         'PCT', # PERCENT
                                         'MON', # MONEY
                                         'QUA', # QUANTITY
                                         'ORD', # ORDINAL
                                         'CAR'] # CARDINAL
            self.model['training_corpus'] = ["?"]
        else:
            raise ValueError("Wrong modelname; must be 'pretrained-spacy',\
                             'pretrained-SENNA', 'pretrained-MITIE',\
                             or 'pretrained-StanfordNER'.")

    def evaluate(self, tgt_test):
        """ Evaluate the model on data tgt_test.

        """
        if self.pretrained_model == 'pretrained-StanfordNER':
            sents = [ [x[0] for (x,iob) in iobs] for iobs in tgt_test]
            predicted = self.NER.tag_sents(sents)
            NERchunks = [BIO_utils.stanfordNE2BIO(NERchunk) for NERchunk in predicted]
            NERchunks = [[((x,x),iob) for (x,iob) in C] for C in NERchunks]
            #TODO test if tgt_test is OK, don't need tgt_test_mod.
            tgt_test_mod = [ [ ((x[0], x[0]), iob) for (x,iob) in iobs] for iobs in tgt_test]
            # Symbols like / are removed sometimes. Fix:
            for jj,chunk in enumerate(NERchunks):
                for ii,x in enumerate(chunk):
                    if x[0] == '':
                        chunk[ii] = (tgt_test_mod[jj][ii],x[1])
            tagset_src = self.model['entity_types']
            E = Evaluator(NERchunks, tgt_test_mod, tagset_src)

        elif self.pretrained_model == 'pretrained-MITIE':
            sents = [ [x[0] for (x,iob) in iobs] for iobs in tgt_test]
            NERchunks = []
            for tagged_sent in sents:
                mitie_entities = self.NER.extract_entities(tagged_sent)
                iobtags = BIO_utils.MITIE_NER_to_iobtags(tagged_sent, mitie_entities)
                NERchunk = BIO_utils.stanfordNE2BIO(iobtags)
                NERchunk = [((x,x),iob) for (x,iob) in NERchunk]
                NERchunks.append(NERchunk)
            tagset_src = self.model['entity_types']
            E = Evaluator(NERchunks, tgt_test, tagset_src)

        elif self.pretrained_model == 'pretrained-SENNA':
            sents = [ [x[0] for (x,iob) in iobs] for iobs in tgt_test]
            NERchunks = self.NER.tag_sents(sents)
            NERchunks = [[((x,x),iob) for (x,iob) in C] for C in NERchunks]
            tagset_src = self.model['entity_types']
            E = Evaluator(NERchunks, tgt_test, tagset_src)

        elif self.pretrained_model == 'pretrained-spacy':
            sents = [ [x[0] for (x,iob) in iobs] for iobs in tgt_test]
            NERchunks = []
            for tagged_sent in sents:
                NERchunk = BIO_utils.spacy_iob(tagged_sent)
                NERchunk = BIO_utils.stanfordNE2BIO(NERchunk) # Need BIO format
                NERchunk = [((x,x),iob) for (x,iob) in NERchunk]
                NERchunks.append(NERchunk)
            tagset_src = self.model['entity_types']
            E = Evaluator(NERchunks, tgt_test, tagset_src)
        else:
            if self.transfer_method == 'pred':
                print('There is no testing to be done here (it was done during training).')
                E = self.predscore
            else:
                print('Tagging the test dataset.')
                test_input_sentences = [zip(*t)[0] for t in tgt_test]
                predicted = self.NER.tag_sents(test_input_sentences)
                tagset_src = self.model['entity_types_src']

                E = Evaluator(predicted, tgt_test, tagset_src)

        return E


