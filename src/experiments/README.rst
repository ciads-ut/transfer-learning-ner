NOTE
----

This directory will contain the results of the transfer learning experiments.

Each subdirectory is for the experiments transferring from CONLL 2003 to
a specific target corpus.  The list of experiments to run is given in a .cfg
configuration file within each subdirectory.

This configuration file specifies the experiments to run via:

- a list of random seeds
- a list of number of training sentences in the target corpus,
- a list of transfer learning methods (CRF-TGT, PRED, PRED-CCA, and variants).

It also specifies the entities ("excluded") which are to be regarded as belonging
only to the source corpus, and which should not be used when calculating the
novel micro and macro scores (i.e. the scores over only the subset of entities
which are in the target but not in source corpus). This is done because
some entities may be labeled differently (e.g. 'person' vs 'PER', or
'LOC' vs 'location').


