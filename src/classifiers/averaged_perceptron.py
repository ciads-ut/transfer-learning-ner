"""
This code was modified from NLTK (nltk/tag/perceptron.py) to include POS tags.

Copyright 2013 Matthew Honnibal
NLTK modifications Copyright 2015 The NLTK Project
This module is provided under the terms of the MIT License.

"""
from nltk.tag.perceptron import AveragedPerceptron
from collections import defaultdict
import random

#TODO add load and save as in the POS one in nltk.tag
class PerceptronNER(object):
    """ Modified from the code in /nltk/tag/perceptron.py, see also
        http://spacy.io/blog/part-of-speech-POS-tagger-in-python/
    """

    def __init__(self, feature_detector, verbose):
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        self.feature_detector = feature_detector

    def tag_sents(self, sentences):

        return [self.tag(sent) for sent in sentences]

    def tag(self, tokens):
        """
        Tags tokenized sequence with iob labels.

        Parameters
        ----------
        tokens : list
            list with entries of form (w,t,d)
        """
        #NOTE:Must be called 'tag'. Must work like the tag method in
        # ClassifierBasedTagger, ie takes a tagged stenence [(w,t,dom)...] and
        # return list [((w,t,d),iob)...]
        output  = []
        history = [] #NOTE: Not using this yet.
        for i in xrange(len(tokens)):
            #iob = self.tagdict.get(word)
            #if not iob:
            feats = self._feat_to_defaultdict(tokens, i, history)
            iob = self.model.predict(feats)
            #end if
            output.append( ((tokens[i][0] , tokens[i][1], tokens[i][2]), iob) )
        return output

    def train(self, sentences, num_iterations): # default iterations: 5
        '''Train a model from sentences.
        ``num_iterations`` controls the number of Perceptron training iterations.

        Parameters
        ----------

        sentences: list
            A list of lists of (toks, iob) tuples where tok is of form (w,pos,dom)

        :param num_iterations: Number of training iterations.
        '''
        self._make_tagdict(sentences)
        self.model.classes = self.classes
        for iter_ in range(num_iterations):
            c = 0
            n = 0
            for sentence  in sentences:
                toks = [tok for tok,iob in sentence]
                iobs = [iob for tok,iob in sentence]
                history = [] #NOTE For now not using this.

                for i, tok in enumerate(toks):
                    #guess = self.tagdict.get(word)
                    #if not guess:
                    feats = self._feat_to_defaultdict(toks, i, history)
                    guess = self.model.predict(feats)
                    self.model.update(iobs[i], guess, feats)
                    #end if
                    c += guess == iobs[i]
                    n += 1
            random.shuffle(sentences)
        self.model.average_weights()

    def _feat_to_defaultdict(self, tokens, index, history):
        """ To change feature format so can be used by NLTK's averaged perceptron.
        Input: features function that takes tokens,index,history and returns
        a dictionary.
        """
        feature_newdict = defaultdict(int)
        features = self.feature_detector(tokens, index, history)

        for k,v in features.items():
            feature_newdict[k+' '+unicode(v)] += 1

        return feature_newdict

    def _make_tagdict(self, sentences):
        '''
        Make a tag dictionary for single-tag words.
        :param sentences: A list of list of (word, tag) tuples.
        '''
        counts = defaultdict(lambda: defaultdict(int))
        for sentence in sentences:
            for word, tag in sentence:
                counts[word][tag] += 1
                self.classes.add(tag)
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
#            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
#            n = sum(tag_freqs.values())
#            if n >= freq_thresh and (mode / n) >= ambiguity_thresh:
            self.tagdict[word] = tag
