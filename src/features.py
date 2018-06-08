"""
This contains the various features that can be used by the sequential
classifiers for entity recognition, and a few utility functions.

Each feature function below takes the following arguments:

    `tokens`  = a POS-tagged sentence [(w1, t1, d1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags (a list)
     NOTE: history is not implemented yet.

Remark
------

For more ideas for features see Nadeau and Sekine, 2007. "A survey of named
entity recognition and classification".

"""
from nltk.stem.snowball import SnowballStemmer
import unicodedata
import re
import numpy as np

def ZhangJohnson(tokens, index, history):
    """
    These features were taken from Zhang and Johnson (2003). They correspond
    to the combination B+D+E+F in their paper.

    Feature 'C' was not included, since
    CRFsuite cannot handle 2nd order CRFS (which can use the i-1 and (i-2)th
    labels) or features of the form label_(i-1) & tok_i.

    Feature 'G' (part-of-speech tags), feature 'H' (chunking tags) and
    features 'I' & 'J' (dictionaries and gazetteers specific to the CONLL 2003
    shared task) are not included.

    References
    ----------

    - Turian J, Rativ L, Bengio Y. Word representations: a simple and general
    method for semi-supervised learning. Proceedings of the 48th annual meeting
    of the association for computational linguistics. Association for
    Computational Linguistics, 2010.

    - Zhang, T. and Johnson D. A robust risk minimization based named entity
    recognition system. Proceedings of the seventh conference on natural
    language learning at HLT-NAACL 2003-Volume 4, Association for Computational
    Linguistics, 2003.

    """

    start_pad2 = ('[START2]', '[START2]','[START2]')
    start_pad1 = ('[START1]', '[START1]','[START1]')
    end_pad2 = ('[END2]', '[END2]','[END2]')
    end_pad1 = ('[END1]', '[END1]','[END1]')
    tokens = [start_pad2, start_pad1] + list(tokens) + [end_pad1, end_pad2]
    history = ['[START2]', '[START1]'] + history
    index += 2 # Shift index to accommodate padding in front of list.

    # Feature set 'B': Tokens in a window of 2
    word = tokens[index][0]
    prevword = tokens[index - 1][0]
    prev2word = tokens[index - 2][0]
    nextword = tokens[index + 1][0]
    next2word = tokens[index + 2][0]

    # Feature set 'D': Initial capitalization of tokens in a window of 2
    capitalized = word[0] == word.capitalize()[0]
    prevcapitalized = prevword[0] == prevword.capitalize()[0]
    prev2capitalized = prev2word[0] == prev2word.capitalize()[0]
    nextcapitalized = nextword[0] == nextword.capitalize()[0]
    next2capitalized = next2word[0] == next2word.capitalize()[0]

    # Feature set 'E': All capitalization, all digitals, or digitals containing
    # punctuation (for center word only).
    allcaps = word.isupper() #word == word.upper()
    all_digits = word.isdigit()
    all_letters = word.isalpha()
    # NOTE: Zhang and Johnson use say "digitals containing punctuations"; we
    # interpret this in the same way as Turian, Ratinov and Bengio (2010)
    # to mean "all non-letters"

    # Feature set 'F': Token prefix (lengths 3 & 4), and suffix (1 - 4)
    prefix3 = word[:3]
    prefix4 = word[:4]
    suffix1 = word[-1:]
    suffix2 = word[-2:]
    suffix3 = word[-3:]
    suffix4 = word[-4:]

    features = {
        'word': word,
        'prevword': prevword,
        'prev2word': prev2word,
        'nextword': nextword,
        'next2word': next2word,
        'capitalized': capitalized,
        'prevcapitalized': prevcapitalized,
        'prev2capitalized': prev2capitalized,
        'nextcapitalized': nextcapitalized,
        'next2capitalized': next2capitalized,
        'allcaps': allcaps,
        'all_digits': all_digits,
        'all_letters': all_letters,
        'prefix3': prefix3,
        'prefix4': prefix4,
        'suffix1': suffix1,
        'suffix2': suffix2,
        'suffix3': suffix3,
        'suffix4': suffix4,
                }

    return features


def wordembeddings_as_features(wordEmbeddings, word2Idx):
    """ Function that returns the feature function with word vectors as
    features. It uses the word embedding both for the current word and for
    the words in a windows of size 2 (as done in Turian et al., 2010).

    wordEmbeddings is the numpy array containing the word vectors;
    word2Idx is the dict mapping each word to the row index.

    Remark
    ------

    Word embeddings were not used as features in the COLING paper.

    """
    def _get_embedding(tok):
        ind = word2Idx.get(tok.lower())
        # NOTE the words in the GloVe embedding list are all lowercase.
        if ind is None: # ie out-of-vocabulary
            # Note: get_word_embeddings makes sure that wordEmbeddings[1]
            # is a random (np.random.uniform(-.25,.25) vector, corresponding
            # to token UNKNOWN_TOKEN.
            vec = wordEmbeddings[1]
        else:
            vec = wordEmbeddings[ind]
        return vec

    def featurefun(tokens, index, history):
        features = {}
        start_pad2 = ('PADDING_TOKEN', '[START2]','[START2]')
        start_pad1 = ('PADDING_TOKEN', '[START1]','[START1]')
        end_pad2 = ('PADDING_TOKEN', '[END2]','[END2]')
        end_pad1 = ('PADDING_TOKEN', '[END1]','[END1]')
        tokens = [start_pad2, start_pad1] + list(tokens) + [end_pad1, end_pad2]
        history = ['[START2]', '[START1]'] + history
        index += 2 # Shift index to accommodate padding in front of list.

        word = tokens[index][0]
        #ind = word2Idx.get(word.lower())
        # NOTE the words in the GloVe embedding list are all lowercase.
        vec = _get_embedding(word)
        for i in range(len(vec)):
            features['vec-'+str(i)]= vec[i]

        prevword = tokens[index - 1][0]
        vec = _get_embedding(prevword)
        for i in range(len(vec)):
            features['prev_vec-'+str(i)]= vec[i]

        prev2word = tokens[index - 2][0]
        vec = _get_embedding(prev2word)
        for i in range(len(vec)):
            features['prev2_vec-'+str(i)]= vec[i]

        nextword = tokens[index + 1][0]
        vec = _get_embedding(nextword)
        for i in range(len(vec)):
            features['next_vec-'+str(i)]= vec[i]

        next2word = tokens[index + 2][0]
        vec = _get_embedding(next2word)
        for i in range(len(vec)):
            features['next2_vec-'+str(i)]= vec[i]

        return features

    return featurefun


def combine_several_featfunctions(featureslist):
    """ Combine lists of features. Given featureslist, a list of feature
    functions, this returns a feature function combining the features in
    each feature function.

    """
    combined_features = {}
    def combined_featurefunc(tokens,index,history):
        for featurefunc in featureslist:
            features = featurefunc(tokens, index, history)
            for feat, value in features.items():
                combined_features[feat]=value
        return combined_features

    return combined_featurefunc

