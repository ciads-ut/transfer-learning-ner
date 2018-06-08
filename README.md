Transfer Learning for Entity Recognition of Novel Classes
=========================================================

This repository contains code, datasets, and results for the paper:

Rodriguez, Caldwell and Liu, "Transfer Learning for Entity Recognition of
Novel Classes". COLING, 2018.

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
directory. CONLL 2003 in particular must be placed in the directory
data/conll2003.  See the README files in the data directory for more
information.

The BiLSTM-CRF experiments require pre-trained word embeddings. We used the
GloVe word embeddings. Download the pretrained Stanford GloVe embeddings
(file glove.6B.100d.txt.gz ) to src/word_embeddings.

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

##BiLSTM-CRF experiments

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

