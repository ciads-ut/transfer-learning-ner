# -*- coding: utf-8 -*-
"""
Utilities for loading pre-trained word embeddings and for obtaining the
vocabulary from the corpus.

"""
import os
import numpy as np
try:
    import cPickle as pkl
except:
    import _pickle as pkl  # Python 3
import gzip
import zipfile


def _get_correct_open(filepath, contentfile=None): #second param is content file name, only used in .zip case
    """ Depending on the file type, will return the correct kind of function
    to open it.

    """
    if filepath[-3:] =='.gz':
        openfun = gzip.open
    elif filepath[-4:] =='.zip':
        # assume the zip only contains one file
        def openfun(filepath, mode): #more is 'r' or 'w'
            if mode =='r':
                zz = zipfile.ZipFile(filepath)
            #if mode =='w':
            # Do this later..
            return zz.open(contentfile, 'r')
    else:
        openfun = open
    return openfun


def get_unique_words(allsents):
    """ Returns set of unique words in the form of a dict.

    Parameters
    ----------

    allsents : list
        The list of sentences.

    """
    allwords = [x[0] for sent in allsents for (x, iob) in sent]
    allwords = [w.lower() for w in allwords] # GloVe requires lowercase
    allwords = set(allwords)
    words = {}
    for word in allwords:
        words[word] = True

    return words


def get_word_embeddings(embeddingsPath, allwords, splittype=" "):
    """ Read in word embeddings from the filesystem, automatically trimming
    them (removing the ones that do not appear in the list of words.

    Parameters
    ----------

    embeddingsPath : str
        filepath for the word embeddings
    allwords : dict
        dict where every key is a word
    splittype : str
        The separation used in the file for pre-trained word embeddings.

    Remark: this function was adapted from:
    https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master/2016-11_Seminar/Session 1 - SENNA/

    """
    word2Idx = {}
    wordEmbeddings = []
    print 'Using word embeddings:', embeddingsPath
    openfile = _get_correct_open(embeddingsPath)
    with openfile(embeddingsPath,'r') as fEmbeddings:
        c = 0
        for line in fEmbeddings:
            c+=1
            split = line.strip().split(splittype)
            if '' in split:
                raise ValueError("Error in spacing of word vector file:"+
                " bad formatting. Consider using different value of splittype.")
            word = split[0]
            # fastText and Yogatama pre-trained embedding starts with a line
            # with 2 elements that we need to ignore:
            if len(split) <= 5:
                print 'Ignoring line '+str(c)
                continue

            if len(word2Idx) == 0: #Add padding+unknown
                word2Idx["PADDING_TOKEN"] = len(word2Idx)
                vector = np.zeros(len(split)-1) #Zero vector for 'PADDING' word
                wordEmbeddings.append(vector)

                word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
                vector = np.random.uniform(-0.25, 0.25, len(split)-1)
                wordEmbeddings.append(vector)

            if word.lower() in allwords:
                vector = np.array([float(num) for num in split[1:]])
                wordEmbeddings.append(vector)
                word2Idx[word] = len(word2Idx)

    wordEmbeddings = np.array(wordEmbeddings)
    return wordEmbeddings, word2Idx


def pkl_save(output_path, list_to_save, name_to_print):
    f = gzip.open(output_path, 'wb')
    for thing in list_to_save:
        pkl.dump(thing, f, -1) # Dump using highest protocol available.
    print name_to_print+" stored in "+output_path
    f.close()

