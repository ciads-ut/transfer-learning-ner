import experiment
import utils
import transferlearning as tl

srcname = 'Wikigold'#'re3d-test'# 'MUC7'#'i2b2-14-train'#'OpeNERHotels-train'#'CONLL03-train'#'MUC7'
tgtname = 're3d' #MITRestaurantCorpus'#'i2b2-14-test' #'CONLL03-test'#'MITRestaurantCorpus' #'OntoNotes5-nw' #'MUC6'

src, tgt = experiment.readdata(srcname, tgtname)
src_data = utils.attach_domain(src, 'src')
tgt_data = utils.attach_domain(tgt, 'tgt')
tgt_train, tgt_test = experiment.split_corpus(tgt_data, train_amount=50)
src = src_data

