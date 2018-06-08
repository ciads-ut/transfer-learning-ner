#Notes on installing third part software

transferlearning.py allows one to use external NER software with pre-trained
models. These need to be installed before they can be used.

Also, note that variable EXTLIB_DIR in transferlearning.py must be changed
to wherever the external libraries are located.

Here are some brief instructions on installing spacy, StanfordNER,
MITIE, and SENNA.

##spacy

The spacy models can be downloaded from:

https://github.com/explosion/spacy-models/releases/tag/en_core_web_md-2.0.0

Then:

```
pip install PATH_TO_SPACY_MODELS/en_core_web_md-2.0.0.tar.gz
python -m spacy link en_core_web_md en_default
```

##Stanford NER

Follow the instructions at https://nlp.stanford.edu/software/CRF-NER.shtml

Make sure the following environment variables are set:

STANFORD_MODELS: should contain the paths to the following:

`english.all.3class.distsim.crf.ser.gz`

`english.conll.4class.distsim.crf.ser.gz`

`english.muc.7class.distsim.crf.ser.gz`

CLASSPATH: should contain:

`stanford-ner-2017-06-09/stanford-ner.jar`


##MITIE


##SENNA

For more information, please see:

https://github.com/nltk/nltk/wiki/Installing-Third-Party-Software
