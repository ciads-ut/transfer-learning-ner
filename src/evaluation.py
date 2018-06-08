"""
This includes functions for the evaluation of the NER model on a test dataset.
These include accuracy (at the tag level), and precision, recall and F1 scores
for each class label. This can be done at the tag level, or at the chunk level
(i.e., the prediction matches the gold annotation when they are both the same
entity and when they overlap completely).

Also included: functions to calculate the micro and macro scores (for
precision, recall and F1), the confusion matrix (at the tag level),
and the set of entities which were true positives, true negatives,
false negatives, and the set of predicted vs actual entities.

"""
from collections import Counter, defaultdict
import nltk
from tabulate import tabulate

import sentence_utils


def get_entities_info(test, tagset_no_iob):
    """ Flattens out the sentences to a long list; then returns a dict, where
    each key is an entity. Every entry in the dict is a list of tuples,
    where each tuple contains:

    - the start and end locations for the first and last word in the entity.
    - the entity name.
    - a tuple containing the tokens comprising the entity.

    Parameters
    ----------

    test: list
        List of sentences
    tagset_no_iob : set
        The set of entity labels (without the IOB prefix)

    Remark
    ------
    This assumes IOB is correct, ie. .. O, I, .. does not happen.

    """
    data = [ [(x[0],iob) for (x,iob) in d] for d in test]
    dataflat = [i for u in data for i in u]

    entities = {}
    for key in tagset_no_iob:
        entities[key] = []
    dataflat.append((u'O',u'O'))
    ent = []

    for i,item in enumerate(dataflat[:-1]):
        if item[1] != 'O':
            if item[1][0] == 'B':
                start_pos = i
                end_pos = i
                ent = [ item[0] ]
            else: # == I
                end_pos += 1
                ent.append(item[0])

            if dataflat[i+1][1][2:] != item[1][2:] or dataflat[i+1][1][0] == 'B':
                entitytype = item[1][2:]
                entities[entitytype].append((start_pos, end_pos,  entitytype, tuple(ent) ) )
    return entities


def fix_i_b(sents, fix_or_check):
    """ Fix IOB-errors.

    Parameters
    ----------

    sents: list
        list of sentences
    fix_or_check: str
        Either 'fix' (corrects errors) or 'check' (checks for errors,
        does not fix).

    """
    warnings = ''
    if fix_or_check == 'check':
        for i, sent in enumerate(sents):
            for j, item in enumerate(sent):
                iob = item[1]
                if iob[:2] == 'I-':
                    if j !=0:
                        if sent[j-1][1]=='O':
                            #print sent[j][0][0], ' '.join([s[0][0] for s in sent]), '\n\n'
                            warnings += "I- is preceded by a O at sentence "+str(i)+", tok "+str(j)+"\n"
                        else:
                            if sent[j-1][1][2:] != iob[2:]:
                                #print sent[j][0][0], ' '.join([s[0][0] for s in sent]), '\n\n'
                                warnings += "I- is preceded by an entity of another category at sentence "+str(i)+", tok "+str(j)+"\n"
                    else:
                        sent[j] = (item[0], 'B-' + iob[2:])
                        warnings += "I- is the beginning of a sentence at sentence "+str(i)+", tok "+str(j)+"\n"

        if warnings != '':
            raise ValueError('Found the following I-B errors:\n\n'+warnings)

    elif fix_or_check == 'fix':
        for i, sent in enumerate(sents):
            for j, item in enumerate(sent):
                iob = item[1]
                if iob[:2] == 'I-':
                    if j !=0:
                        if sent[j-1][1]=='O':
                            #print 'a1', sent[j-1][1], sent[j][1]
                            sent[j] = (item[0],'B-' + iob[2:])
                            #print 'a2', sent[j-1][1], sent[j][1]
                            warnings += "I- is preceded by a O at sentence "+str(i)+", tok "+str(j)+"\n"
                        else:
                            if sent[j-1][1][2:] != iob[2:]:
                                #print 'b1', sent[j-1][1], sent[j][1]
                                sent[j] = (item[0],'B-' + iob[2:])
                                #print 'b2', sent[j-1][1], sent[j][1]
                                warnings += "I- is preceded by an entity of another category at sentence "+str(i)+", tok "+str(j)+"\n"
                    else: # Beginning of sentence should not start with I
                        sent[j] = (item[0], 'B-' + iob[2:])
                        warnings += "I- is the beginning of a sentence at sentence "+str(i)+", tok "+str(j)+"\n"

        if warnings != '':
            print("Fixed the following I-B errors:\n\n"+warnings)

    else:
        raise ValuError("Use 'fix' or 'check'.")


def sent2line(s, ii,tgt_test):
    ind = len([i for u in tgt_test[0:s] for i in u]) + ii
    return ind


class Evaluator():
    """ This class is used to evaluate the entity extractor.

    Remark
    ------

    How this relates to the NLTK chunk scorer:

    |allents_actual| = |score.correct| = TP + FN
    |allents_pred| = |score.guessed| = TP + FP
    FN = |score.missed|

    """
    def __init__(self, predicted, actual, tagset_src):
        #NOTE In case some sentences are too long, first check the lengths match,
        # and if they don't, crop the actual to fit the predicted.
        #NOTE for the experiments run this only applies to i2b2-14
        assert len(predicted) == len(actual)
        for ii, sent in enumerate(actual):
            predlen = len(predicted[ii])
            if len(sent) != predlen:
                print 'Sentence was too long and prediction was truncated at sent. #', ii
                print '\n\nFor purpose of evaluation, going to truncate the actual sentence.\n\n'
                actual[ii] = sent[:predlen]

        self.predicted = predicted
        self.tagset_src = set(tagset_src)

        # Before doing anything else, will replace all I- with B- entities.
        fix_i_b(self.predicted,'fix')
        self.actual = actual
        xpred = [iob for p in self.predicted for (x,iob) in p]
        xact = [iob for p in self.actual for (x,iob) in p]
        print('Obtaining the class label set.')
        self.tagset = set(xact).union(set(xpred))
        self.tagset_no_iob = set([t[2:] for t in list(self.tagset) if t != 'O'])

        # do this way.. .TODO remove the other way.
        assert self.tagset == sentence_utils.get_tagset(self.predicted + self.actual, with_prefix = True)
        assert self.tagset_no_iob == sentence_utils.get_tagset(self.predicted + self.actual, with_prefix = False)

        print('Obtaining actual and predicted entity information.')
        self.ents_actual = get_entities_info(self.actual, self.tagset_no_iob)
        self.ents_pred = get_entities_info(self.predicted, self.tagset_no_iob)

        self.predflat = [(x[0][0],x[1]) for u in self.predicted for x in u]
        self.actflat = [(x[0][0],x[1]) for u in self.actual for x in u]

    def _XX_set(self, set_operation, entity = 'all'):
        if entity != 'all' and entity not in self.tagset_no_iob:
            raise ValueError("entity must be one of "+str(self.tagset_no_iob)[5:-2])
        if entity == 'all':
            allents_actual = [item for i in self.tagset_no_iob for item in self.ents_actual[i] ]
            allents_pred = [item for i in self.tagset_no_iob for item in self.ents_pred[i] ]
            XX_set = set_operation( set(allents_actual), set(allents_pred) )
        else:
            XX_set = set_operation( set(self.ents_actual[entity]), set(self.ents_pred[entity]) )

        return XX_set

    def TP_set(self, entity='all'):
        """ The set of true positives.

        """
        TP_set = self._XX_set(set.intersection, entity = entity)
        return TP_set

    def FN_set(self, entity='all'):
        """ The misses: cases where the model didn't predict an entity.
        The entity type displayed is the actual entity type.

        """
        FN_set = self._XX_set(set.difference, entity = entity)
        return FN_set

    def FP_set(self, entity='all'):
        """ Those which are incorrectly labeled as entity. The entity type
        displayed is the predicted entity type.

        Entities could be in this set either because they are of the incorrect
        type or because the text spans did not match, or both.

        """
        setdiff_rev = lambda L,D: set.difference(D,L)
        FP_set = self._XX_set(setdiff_rev, entity = entity)
        return FP_set

    def confusion_matrix(self):
        """ Confusion matrix at the token level. The row index is the actual
        value; the column is the predicted value.

        """
        xpred = [iob for p in self.predicted for (x, iob) in p]
        xact = [iob for p in self.actual for (x, iob) in p]

        self.cm = nltk.ConfusionMatrix(xact, xpred)
        #print(self.cm.pretty_format())
        return self.cm

    def accuracy(self, chunk_or_tag='tag'):
        """ The fraction of tokens that were correctly classified (at the tag
        level). This matches NLTK's ChunkScore accuracy.

        """
        if chunk_or_tag == 'chunk':
            raise ValueError("Accuracy does not make sense at the chunk level. Use 'tag'.")

        cm = self.confusion_matrix()
        accuracy = float(sum([cm[i,i] for i in self.tagset]))/sum([cm[i,j] for i in self.tagset for j in self.tagset])
        return accuracy

    def precision(self,  chunk_or_tag = 'chunk'):
        """ Compute the precision, at either the tag level or the chunk level.

        """
        if chunk_or_tag == 'tag':
            cm = self.confusion_matrix()
            precisions = {}
            for etype in self.tagset:
                TP_plus_FP = sum([cm[i, etype] for i in self.tagset])
                if TP_plus_FP !=0:
                    precisions[etype] = float(cm[etype,etype])/TP_plus_FP
                else:
                    precisions[etype] = None
        elif chunk_or_tag == 'chunk':
            precisions = {}
            for etype in self.tagset_no_iob:
                TP = len(self.TP_set(etype))
                FP = len(self.FP_set(etype))
                if TP + FP != 0:
                    precisions[etype] = float(TP)/(TP+FP)
                else:
                    precisions[etype] = None
        else:
            raise ValueError("Use 'chunk' or 'tag'.")
        return precisions

    def recall(self, chunk_or_tag = 'chunk'):
        """ Compute the recall, at either the tag level or the chunk level.

        """
        if chunk_or_tag == 'tag':
            cm = self.confusion_matrix()
            recalls = {}
            for etype in self.tagset:
                TP_plus_FN = sum([cm[etype, i] for i in self.tagset])
                if TP_plus_FN !=0:
                    recalls[etype] = float(cm[etype,etype])/TP_plus_FN
                else:
                    recalls[etype] = None
        elif chunk_or_tag == 'chunk':
            recalls = {}
            for etype in self.tagset_no_iob:
                TP = len(self.TP_set(etype))
                FN = len(self.FN_set(etype))
                if TP + FN != 0:
                    recalls[etype] = float(TP)/(TP+FN)
                else:
                    recalls[etype] = None
        else:
            raise ValueError("Use 'chunk' or 'tag'.")

        return recalls

    def F1(self, chunk_or_tag = 'chunk'):
        """ Compute the F1 score, at either the tag level or the chunk level.

        """
        if chunk_or_tag not in {'tag','chunk'}:
            raise ValueError("Use 'chunk' or 'tag'.")

        P = self.precision(chunk_or_tag = chunk_or_tag)
        R = self.recall(chunk_or_tag = chunk_or_tag)
        F1s = {}
        tagset = self.tagset if chunk_or_tag == 'tag' else self.tagset_no_iob
        for etype in tagset:
            if P[etype] is None or R[etype] is None:
                F1s[etype] = None
            elif P[etype] == 0 and R[etype] == 0:
                F1s[etype] = 0
            else:
                F1s[etype] = 2*P[etype]*R[etype] / (P[etype] + R[etype])
        return F1s

    def microPRF1(self, chunk_or_tag = 'chunk', labelset = 'all', include_O=False):
        """ Compute the micro-averaged precision, recall and F1 scores.

        microF1 with 'chunk' should coincide with NLTK's score.f_measure()

        Remark
        ------

        Note: include_O only applies in the 'tag' case. In the 'chunk' case
        O is not included in the average.

        Notes on microF1 with 'tag':
            1) if include_O = True, this should coincide with
               sklearn.metrics.f1_score(xact,xpred, average='micro')
            2) FN and FP are the same size. Does this scenario make sense?

        """
        if chunk_or_tag == 'chunk':
            if labelset == 'all':
                TP_set = self.TP_set()
                FN_set = self.FN_set()
                FP_set = self.FP_set()

                assert set([item for i in self.tagset_no_iob for item in self.ents_actual[i] ]) == TP_set.union(FN_set)
                assert set([item for i in self.tagset_no_iob for item in self.ents_pred[i] ]) == TP_set.union(FP_set)

                TP = len(TP_set)
                FN = len(FN_set)
                FP = len(FP_set)
            elif isinstance(labelset, set) or isinstance(labelset, list):
                if not set(labelset).issubset(self.tagset_no_iob):
                    raise ValueError("labelset must be a subset of existing labels.")
                TP = sum([ len(self.TP_set(entity=tag)) for tag in labelset])
                FN = sum([ len(self.FN_set(entity=tag)) for tag in labelset])
                FP = sum([ len(self.FP_set(entity=tag)) for tag in labelset])
            else:
                raise ValueError("labelset must be 'all' or a set of labels.")

        elif chunk_or_tag == 'tag':
            if labelset != 'all':
                raise ValueError("'tag' with arbitrary label sets are not implemented yet.")
            cm = self.confusion_matrix()
            if include_O:
                TP = sum([cm[i, i] for i in self.tagset])
                FN = sum([ sum([cm[j, i] for i in self.tagset]) - cm[j,j] for j in self.tagset])
                FP = sum([ sum([cm[j, i] for i in self.tagset]) - cm[j,j] for j in self.tagset])
            else:
                tagset = self.tagset - {'O'}
                TP = sum([cm[i, i] for i in tagset])
                FN = sum([ sum([cm[j, i] for i in tagset]) - cm[j,j] for j in tagset])
                FP = sum([ sum([cm[j, i] for i in tagset]) - cm[j,j] for j in tagset])
        else:
            raise ValueError("Use 'chunk' or 'tag'.")

        P = float(TP)/(TP+FP) if TP+TP !=0 else 0
        R = float(TP)/(TP+FN) if TP+FN !=0 else 0
        F1 = 2*P*R/ (P + R) if P+R !=0 else 0

        return P, R, F1

    def microPRF1_noveltypes(self, more_excluded_entities):
        """ Evaluate chunk-level micro Precision, Recall and F1, but only over
        novel entity types, i.e., entity types in the target corpus that are
        not present in the source corpus.

        In the event that the same entity type is labeled differently in
        source and target (e.g., "LOC" in the source, "geo-loc" in the target),
        one may specify the entities in the target set which should be excluded
        as well in the set 'more_excluded_entities').

        Only applies to chunk-level evaluation (for the moment).

        """
        labelset = self.tagset_no_iob - self.tagset_src
        self.tagset_novel = labelset - more_excluded_entities
        if self.tagset_novel != set():
            res = self.microPRF1(chunk_or_tag = 'chunk',
                                 labelset = self.tagset_novel,
                                 include_O = False)
        else:
            res = False
        return res

    def macroPRF1_noveltypes(self, more_excluded_entities):
        """ Evaluate chunk-level macro Precision, Recall and F1, but only over
        novel entity types, i.e., entity types in the target corpus that are
        not present in the source corpus.

        In the event that the same entity type is labeled differently in
        source and target (e.g., "LOC" in the source, "geo-loc" in the target),
        one may specify the entities in the target set which should be excluded
        as well in the set 'more_excluded_entities').

        Only applies to chunk-level evaluation (for the moment).

        """
        labelset = self.tagset_no_iob - self.tagset_src
        self.tagset_novel = labelset - more_excluded_entities
        if self.tagset_novel != set():
            res = self.macroPRF1(chunk_or_tag = 'chunk',
                                 labelset = self.tagset_novel,
                                 include_O = False)
        else:
            res = False
        return res

    def macroPRF1(self, chunk_or_tag = 'chunk', labelset = 'all', include_O=False):
        """ Calculate the macro-average of the F1, precision, and recall
        scores for each class.

        labelset is either 'all', or a set of labels over which to average.

        macro-F1 with 'tag':
               If include_O = True, this should coincide with
               sklearn.metrics.f1_score(xact,xpred, average='macro')

        Note: include_O only applies in the 'tag' case. In the 'chunk' case
        O is not included in the average.

        Note: if the F1 score for any given class label is undefined (None)
        then we set it to zero and include it in the average
        (sklearn does this too).

        Why this is appropriate: being None means that either TP+FP or
        TP + FN was 0. In either case, this means TP = 0.
        There are 3 cases:
        R=None and P=0, R=0 and P=None, or both R and P are None.
        In the first two cases the F1 score should be 0.
        The third case is problematic.
        It means there are no instances with the given label and no
        predictions of that label. This is probably unlikely, but if it
        happens, we should probably not penalize the macro-F1 by setting the None
        to 0 ... For now, just output error.

        """
        F1s = self.F1(chunk_or_tag = chunk_or_tag)
        P = self.precision(chunk_or_tag = chunk_or_tag)
        R = self.recall(chunk_or_tag = chunk_or_tag)

        for k,v in P.iteritems():
            if v==None and R[k] ==None:
                raise ValueError("There are no predictions and no labels for some class")
        if labelset == {}:
            raise ValueError("The labelset is empty.")

        if chunk_or_tag == 'tag':
            if not isinstance(labelset,str) and not set(labelset).issubset(self.tagset):
                raise ValueError("labelset must be a subset of the tagset")
            if not include_O: # remove the 'O' class [Note: if 'chunk', O is already removed.]
                F1s.pop('O')
                P.pop('O')
                R.pop('O')
                if not isinstance(labelset,str) and 'O' in labelset:
                    raise ValueError("'O' cannot be in the labelset if include_O is False")
        elif chunk_or_tag == 'chunk':
            if not isinstance(labelset,str) and not set(labelset).issubset(self.tagset_no_iob):
                raise ValueError("labelset must be a subset of the tagset_no_iob")
            if not isinstance(labelset,str) and 'O' in labelset:
                raise ValueError("'O' cannot be in the labelset when using 'chunk' scores.")
        else:
            raise ValueError("Use 'chunk' or 'tag'.")

        if labelset == 'all':
            macro_F1 = sum([F1s[k] for k in F1s if F1s[k] is not None])/len(F1s)
            macroP = sum([P[k] for k in P if P[k] is not None])/len(P)
            macroR = sum([R[k] for k in R if R[k] is not None])/len(R)
        else:
            macro_F1 = sum([F1s[k] for k in F1s if k in labelset and F1s[k] is not None])/len(labelset)
            macroP = sum([P[k] for k in P if k in labelset and P[k] is not None])/len(labelset)
            macroR = sum([R[k] for k in R if k in labelset and R[k] is not None])/len(labelset)

        return macroP, macroR, macro_F1

    def write_report(self, filename, more_excluded_entities):
        """ Write a summary of evaluation statistics, including macro- and
        micro-averaged recall, precision and F1 scores, per-class scores,
        micro- and macro- scores over novel classes, and a confusion matrix
        at the token level.

        """
        if more_excluded_entities is None:
            more_excluded_entities = set()

        with open(filename, 'a+') as fd:
            fd.write('Results for NER\n===============\n\n')

            fd.write('Micro and macro-averaged Precision, Recall, and F1 scores:\n\n')
            avgPRF1 = [['Micro-avg']+list(self.microPRF1() ),
                       ['Macro-avg']+list(self.macroPRF1() )]
            fd.write(tabulate(avgPRF1,
                              headers = ['','P','R','F1'],
                              missingval="None", floatfmt=".4f",
                              numalign='left'))
            fd.write('\n\n')

            macroPRF1_novel = self.macroPRF1_noveltypes(more_excluded_entities)
            microPRF1_novel = self.microPRF1_noveltypes(more_excluded_entities)

            if macroPRF1_novel and microPRF1_novel:
                fd.write('Micro and macro-averaged Precision, Recall, and F1 scores over label subset:\n\n')
                avgPRF1_novel = [['Micro-avg']+list(microPRF1_novel),
                                 ['Macro-avg']+list(macroPRF1_novel) ]
                fd.write(tabulate(avgPRF1_novel,
                                  headers = ['','P','R','F1'],
                                  missingval="None", floatfmt=".4f",
                                  numalign='left'))
                fd.write('\n\nAveraged over the following labels:\n\n')
                fd.write('\n'.join(sorted(list(self.tagset_novel))) )
                fd.write('\n\n')

            fd.write('Precision, Recall, and F1 scores for all class labels:\n\n')
            PRF1 = [ [k,
                    self.precision()[k],
                    self.recall()[k],
                    self.F1()[k]]
                    for k in sorted(list(self.tagset_no_iob))]
            fd.write(tabulate(PRF1,
                              headers = ['Class','P', 'R', 'F1'],
                              missingval="None", floatfmt=".4f",
                              numalign='left'))
            fd.write('\n\n')

            fd.write('Confusion matrix (token-level):\n\n')
            fd.write(self.confusion_matrix().pretty_format())
            fd.write('\n\n')

