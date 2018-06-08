""" Utilities related to IOB-tag entity conversion useful when using other
entity extraction tools with pre-trained models: spacy, StanfordNER, or MITIE.

"""

import os
from nltk.data import path as nltk_data_path
nltk_data_location = os.getenv('NLTK_DATA_PATH')
if nltk_data_location is not None:
    nltk_data_path.append(nltk_data_location)
from nltk import pos_tag
from nltk.chunk import conlltags2tree
from nltk.tree import Tree

import spacy


def LONGENT_to_SHORTENT(entityname):
    """ Shortens the entity label name.

    """
    if entityname in {'O','MISC','GPE','GSP'}:
        shortentity = entityname
    elif entityname == 'PERCENT':
        shortentity = u'PCT'
    else: # convert to LOC, PER, ORG, etc
        shortentity = entityname[:3]
    return shortentity


def spacy_iob(sent):
    """ sent is a list of tokens in the sentence.

    For more information, see: https://spacy.io/docs/usage/entity-recognition

    """
    nlp = spacy.load('en_default')
    text = unicode(' '.join(sent))
    doc = nlp(text)
    ents = doc.ents
    iob_tags = ['O']*len(sent)

    for ent in ents:
        chunk = text[ent.start_char:ent.end_char]
        chunkwords = chunk.split()
        ent_type = ent.label_
        inds = [i for i,w in enumerate(sent) if w in chunkwords]
        for i in inds:
            iob_tags[i] = ent_type

    return zip(sent,iob_tags)


def MITIE_NER_to_iobtags(sent, mitie_ents):
    """ Used to convert MITIE entities."""
    # sent is the list of words
    #return list of (w, iob); later can use the add_pos to this.
    iobs = ['O']*len(sent)
    for ent in mitie_ents:
        for i in ent[0]:
            iobs[i]=ent[1]
    return zip(sent,iobs)


def stanfordNE2BIO(tagged_sent):
    """ Transforms the output of Stanford NER to BIO format.
    Also changes LOCATION, PERSON, etc. labels to LOC, PER, etc

    """
    for i, (token, tag) in enumerate(tagged_sent):
        tagged_sent[i] = (token, LONGENT_to_SHORTENT(tag))

    bio_tagged_sent = []
    prev_tag = "O"
    for token, tag in tagged_sent:
        if tag == "O": #O
            bio_tagged_sent.append((token, tag))
            prev_tag = tag
            continue
        if tag != "O" and prev_tag == "O": # Begin NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag: # Inside NE
            bio_tagged_sent.append((token, "I-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag: # Adjacent NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag

    return bio_tagged_sent


def add_pos_tags(ne_tagged_sent):
    """ne_tagged sent is a list of (w,iob) where w is the token, and
    iob is the NER label. This returns a list of (w,pos,iob), where the pos
    is obtained using NLTK's pos_tag.

    """
    tokens, iob_tags  = zip(*ne_tagged_sent)
    pos_tags = [p for w, p in pos_tag(tokens)]
    iob_triplets = zip(tokens, pos_tags, iob_tags)
    #ne_tree = conlltags2tree(sent_conlltags)
    return iob_triplets

