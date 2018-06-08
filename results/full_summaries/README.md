# Results

This directory contains results.txt files containing the following
information for each experiment run:

- Precision, recall and F1 scores for all class labels.
- Micro-averaged scores (precision, recall, F1).
- Macro-averaged scores (precision, recall, F1).
- Micro and macro averaged scores over novel labels.
- Confusion matrix (at the token level).

# Note on directory structure

Every subdirectory contains the following directory structure. The
following are nested:

- seed_X: seed number, subdirectory for all experiments with a given seed X
  (for shuffling the target dataset).
- tgt_train_len_X: subdirectory for experiments run with X number of sentences
  in the target training dataset.
- method_X: subdirectory for the transfer method, where X can be:
    - tgt: CRF training on the target training set only.
    - pred-no_prefix: method PRED, without IOB prefixes.
    - predCCA-no_prefix-excludeO: method PRED-CCA, without IOB prefix, and
      excluding the 'O' entity label from the CCA calculation.
    - predCCA-no_prefix: method PRED-CCA without IOB prefix.
    - pretrainEL_rmsprop_v3sgd: BiLSTM-CRF trained on CONLL 2003, fine-tuned
      on the target corpus, and fine-tuning both the embedding layer and the
      LSTM layer.
    - pretrainN_rmsprop_v3sgd: BiLSTM-CRF trained from scratch on the target
      corpus (no pre-training).

