Transfer Learning for Entity Recognition of Novel Classes
=========================================================

This repository contains code, datasets, and results for the paper:

Rodriguez, Caldwell and Liu, "Transfer Learning for Entity Recognition of
Novel Classes". COLING, 2018.

This paper can be downloaded from http://aclweb.org/anthology/C18-1168

In addition to this, it includes:

- An API to evaluate different combinations of classifiers and transfer
  learning methods. See the documentation in transferlearning.py for more
  details.
- A wrapper for other NER tools (pre-trained models), including SENNA, MITIE,
  StanfordNER and spacy. To install these, see THIRD_PARTY_SOFTWARE.rst
- Utility functions for handling datasets that could not be distributed.

# Set up

This code was tested and run on Linux, using Python 2. The list of required
Python packages is in requirements.txt. In addition, keras-contrib
(2.0.8) needs to be installed. keras-contrib2.0.8.tar.gz is included
in this repository. Extract the contents of keras-contrib2.0.8.tar.gz and
run the following to install keras-contrib:

`python setup.py install`

To run the experiments, first make sure all the datasets are in the appropriate
directory. The files for the CONLL 2003 dataset (eng.train, eng.testa,
eng.testb) in particular must be placed in directory `data/conll2003`.
See the README files in the data directory for more information.

The BiLSTM-CRF experiments require pre-trained word embeddings. We used the
GloVe word embeddings. Download the pretrained Stanford GloVe embeddings
from http://nlp.stanford.edu/data/glove.6B.zip and put file
`glove.6B.100d.txt.gz` in `src/word_embeddings`.

# Datasets

Due to licensing restrictions, GUM and re3d were the only datasets
from our paper that could be included in this repository.
Since each of the datasets comprising re3d
has a different license, the train/test split of re3d used in our paper
is not included. However, it can be easily generated following the
directions in `data/re3d/CONLL-format/data/README.md`.

Instructions for obtaining the other datasets used in the paper are found
in each of the corresponding dataset directories, together with directions
for where to place them. The file locations will correspond to those listed
in the file `src/file_locations.cfg`. Ritter's Twitter dataset, the MIT Movie
Corpus and the MIT Restaurant Corpus can be downloaded and are already in the
CONLL 2003 format. The remaining datasets are in different
formats; tools are included to convert them to the CONLL 2003 format.

In addition, we include instructions for obtaining several other NER
datasets not used in the COLING paper, but which may be of interest. These
are:

- BBN
- MalwareTextDB
- GMB 1.0.0
- Wikigold
- AnEM
- WNUT 2017
- SEC-filings

Of these, only Wikigold, AnEM, WNUT 2017 and SEC-filings could be shared due to
licensing restrictions.

# Dataset licenses

For a summary of the dataset licenses, see `data/LICENSES_SUMMARY.rst`. Each
data directory also includes the license for that dataset.

# Reproducing the experiments

The directory src/experiments contains several subdirectories of the form
CONLL03_to_X, where X is the name of the target corpus. Each of these
subdirectories contains a .cfg configuration file specifying which parameters
to use. These include source corpus, target corpus, random seeds,
number of training sentences, algorithms to evaluate, and transfer learning
method. The transfer learning methods do not include the neural methods
(these are run separately because they take more time).

At the moment, the experiments must be run in the following order:

- First run the CRF-based experiments.
- Then the BiLSTM-CRF experiments can be run.

## CRF experiments

To run the CRF experiments (CRF-TGT, PRED, and PRED-CCA), run for example:

```
import experiment
experiment.run_experiment('CONLL03_to_GUM')
```

This will run all the specified experiments from the configuration file and
save the results in directory CONLL03_to_GUM. The results will contain both
the scores for each run (in a results.txt file) and the raw predictions as
well (in file predicted.conll), as well as the macro- and micro- averaged
results in a pkl file, which can be loaded through pandas.

## BiLSTM-CRF experiments

Once the CRF experiments have been run, one may run the BiLSTM-CRF experiments.
This is done in two steps. First, train the network on the source (CONLL 2003)
corpus (this only needs to be done once, since the vocabulary for the word
embeddings is the union of the vocabulary of the source and the vocabulary
of each possible target dataset). This is done via:

```
import train_bilstm_model as tbm
max_len, we, w2i, words = tbm.get_embeddings()
history, score = tbm.fit_and_test_model(max_len, we, w2i, words)
```

Then, to fine-tune the neural network, run:

```
import load_pretrained
load_pretrained.make_reports(tgt_corpus, 'rmsprop_v3sgd', tlayers)
```

where `tgt_corpus` is 'GUM', 'TwitterRitter', 're3d', 'NIST99', 'MUC6',
'MITRestaurant', 'MITMovie', 'i2b2-14', 'i2b2-06' or 'CADEC', and
`tlayers` can be 'N' (no transfer, train from scratch), 'E' (transfer the
embedding layer only) or 'EL' (transfer both the embedding layer and
biLSTM layer).

To replicate the results of the paper, run the code using both 'N' and 'EL'.
This will create files results_pretrainEL_rmsprop_v3sgd.pkl and
results_pretrainN_rmsprop_v3sgd.pkl in the appropriate directory. These can
be opened with pandas, and contain the averaged scores for these runs.

In addition, the scores for each run will be saved in a results.txt file,
and the raw predictions will also be saved in predicted.conll files.

