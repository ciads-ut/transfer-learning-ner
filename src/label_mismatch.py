""" Functions for the label mismatch problem, using CCA to obatin label
    embeddings. This also includes some functions for visualizing the
    embeddings.

"""
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import manifold
import matplotlib.pyplot as plt
#from adjustText import adjust_text
from mpl_toolkits.mplot3d import Axes3D

import sentence_utils

def cca(sents,
        no_prefix = False,
        k=5,
        exclude_O = False,
        normalize_rows = True,
        plotlabels = False):
    """ Calculate the label embeddings through CCA, as done in the paper:

    Kim et al., 2015. "New transfer learning techniques for disparate label
    sets"

    Parameters
    ----------

    sents : list
        List of sentences.
    no_prefix : bool
        Whether or not to do label embeddings for the labels without an
        IOB prefix (if no_prefix=True) or with a prefix (if no_prefix=False).
    exclude_O : bool
        Whether or not to exclude the O 'entity' from the CCA calculation.
        If O is included, 'O' will be mapped to the zero vector.
    normalize_rows : bool
        Whether or not to normalize the rows.
    plotlabels : bool
        Whether or not to make a plot of the labels. If this is True, it will
        only make a plot if k=2 (in two dimensions) or k=3 (in three
        dimensions).

    Returns
    -------

    label2vec : dict
        Dictionary mapping class labels to k-dimensional vectors.

    """
    if no_prefix:
        sents = sentence_utils.sents_no_prefix(sents)

    label2ind, ind2label, word2ind, ind2word, ner_tag_count, word_count, counts, normalized_counts = get_stats(sents, exclude_O = exclude_O)

    numtags = len(ner_tag_count)
    if k >= numtags:
        raise ValueError("k must be less than the number of tags: "+str(numtags))

    U, S, V = np.linalg.svd(normalized_counts, full_matrices=False)
    newU = cut_and_renormalize_rows(U, k, normalize_rows = normalize_rows)
    # each row corresponds to a label.
    label2vec = {}
    for label in label2ind.keys():
        label2vec[label] = newU[label2ind[label], :]

    # If we excluded O from the CCA calculation, we need to set an embedding
    # value for it. For now use the 0 vector.
    if exclude_O:
        O_labelvec = np.zeros(k)
        label2vec['O'] = O_labelvec

    if k == 2 and plotlabels:
        if exclude_O:
            X = np.append(newU[:,0],0)
            Y = np.append(newU[:,1],0)
        else:
            X, Y = newU[:,0], newU[:,1]
        plt.scatter(X, Y, s = 6)
        plt.axis('equal')

        texts = [plt.text(newU[i,0], newU[i,1], ind2label[i], size=7) for i in range(newU.shape[0])]
        #adjust_text(texts, arrowprops = dict(arrowstyle = "-", color = "k",
        #            lw = 0.5))

        plt.grid()

    if k == 3 and plotlabels:
        if exclude_O:
            X = np.append(newU[:,0],0)
            Y = np.append(newU[:,1],0)
            Z = np.append(newU[:,2],0)
        else:
            X, Y, Z = newU[:,0], newU[:,1], newU[:,2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(X, Y, Z, s = 6, marker = 'o')
        ax.axis('equal')

        texts = [ax.text(newU[i,0], newU[i,1], newU[i,2], ind2label[i], size=7)
                for i in range(newU.shape[0])]

    return label2vec


def get_stats(sents,
              exclude_O = False):
    """ Returns the statistics required by CCA from the list of sentences.
    This includes matrices of word-entity co-occurrence counts, word counts,
    and entity counts.

    Parameters
    ----------

    sents : list
        List of sentences.
    exclude_O : bool
        Whether or not to exclude the O 'entity' from the CCA calculation.
        If O is included, 'O' will be mapped to the zero vector.

    """
    f = sentence_utils.flatten(sents)
    print('Tokens: '+str(len(f)))

    ner_tag_count = sentence_utils.get_IOB_counts(sents)
    #print('Label frequency:')
    #for i in ner_tag_count:
    #    print(i, ner_tag_count[i])
    if exclude_O:
        ner_tag_count.pop('O')

    labels = list(ner_tag_count)
    print('Label dim: '+str(len(labels)))

    word_count = sentence_utils.get_word_counts(sents, exclude_O = exclude_O)
    vocabulary = word_count.keys()
    print('Vocabulary size: '+ str(len(vocabulary)))

    label2ind, ind2label, word2ind, ind2word = _mappings(labels, vocabulary)

    words = [x[0] for (x, iob) in f]
    iobs = [iob for (x, iob) in f]

    counts = np.zeros((len(labels), len(vocabulary) ))
    normalized_counts = np.zeros((len(labels), len(vocabulary) ))

    if exclude_O:
        for ind, w in enumerate(words):
            if iobs[ind] != 'O':
                i = label2ind[iobs[ind]]
                j = word2ind[w]
                counts[i,j] += 1
    else:
        for ind, w in enumerate(words):
            i = label2ind[iobs[ind]]
            j = word2ind[w]
            counts[i,j] += 1

    for i in xrange(len(labels)):
        for j in xrange(len(vocabulary)):
            count_l = ner_tag_count[ind2label[i]]
            count_w = word_count[ind2word[j]]
            normalization = np.sqrt(count_l * count_w)
            normalized_counts[i,j] = counts[i,j]/normalization

    return label2ind, ind2label, word2ind, ind2word, ner_tag_count, word_count, counts, normalized_counts


def _mappings(labels, words): #list of labels
    label2ind = {}
    word2ind = {}
    ind2label = {}
    ind2word = {}

    for i,x in enumerate(labels):
        label2ind[x] = i
        ind2label[i] = x
    for i,x in enumerate(words):
        word2ind[x] = i
        ind2word[i] = x

    return label2ind, ind2label, word2ind, ind2word


def kmeans(U, ind2label, num_clusters=8):
    """ Each now of U is an observation vector; this will cluster them.

    """
    kmeans = KMeans(n_clusters=num_clusters).fit(U)
    clusters = kmeans.predict(U)
    cluster_list = []

    for cluster in range(num_clusters):
        ww = [i for i,x in enumerate(clusters) if x==cluster]
        cluster_list.append([ind2label[w] for w in ww])

    return clusters, cluster_list


def cut_and_renormalize_rows(U, k, normalize_rows=True):
    """ If k=d, then all the rows of U are equidistant; this does not make
    sense for the application where each row of U is a label vector.

    Instead, if k<d, can use same matrix U, but just use the first k columns,
    and renormalize.

    """
    newU = U[:,0:k]
    if normalize_rows:
        for i in range(U.shape[0]):
            newU[i] = newU[i]/norm(newU[i])
    return newU


def plot_TSNE_summary(normalized_counts,
                      i2l,
                      embedding='MDS',
                      normalize_rows = True):
    """ View 2-D embedding (using MDS, LLD or TSNE) of the label embeddings,
    given the normalized_counts matrix.

    This displays a grid of plots, variying both (a) the number of clusters
    for the kmeans clustering, and (b) the value of k (the label embedding
    dimension).

    Parameters
    ----------

    normalized_counts: numpy array
        Count matrix used to obtain label embeddings via CCA.

    i2l : dict
        Mapping of the index to the label name.

    embedding : str
        'MD5', 'TSNE' or 'LLE'

    normalize_rows : bool
        Whether or not to normalize the rows.

    """
    U, S, V = np.linalg.svd(normalized_counts, full_matrices=False)
    #fig = plt.figure()
    numclusters = [3,6]  #[2,6] # Number of rows to display.
    krange = [5,9]  #[3,6,9,11] # Number of columns to display.
    ncols = len(krange)
    nrows = len(numclusters)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)

    n_neighbors = 4
    plotnum=0
    for jj,k in enumerate(krange):
        newU = cut_and_renormalize_rows(U, k, normalize_rows = normalize_rows)
        for ii,numcluster in enumerate(numclusters):
            plotnum += 1
            clusters, cl = kmeans(newU, i2l, numcluster)
            clustercolors = [i/(newU.shape[0]+0.0) for i in clusters]
            cm = plt.cm.get_cmap('RdYlBu')

            if embedding=='MDS':
                print("Computing MDS embedding")
                clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
                emb = clf.fit_transform(newU)
            elif embedding=='TSNE':
                model = TSNE(n_components=2, init='pca')
                tsne = model.fit_transform(newU)
                avg_norm = np.mean([norm(tsne[i]) for i in range(tsne.shape[0]) ])
                tsne = tsne/avg_norm
                emb = tsne
            elif embedding=='LLE':
                clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                                              method='standard')
                emb = clf.fit_transform(newU)

            else:
                raise ValueError("Wrong embedding name.")
            axs[ii,jj].scatter(emb[:,0],emb[:,1], c=clustercolors, s=35,cmap=cm)
            for i in range(emb.shape[0]):
                axs[ii,jj].text(emb[i,0], emb[i,1], i2l[i])
            axs[0, jj].set_title('Embedding dimension k: %d' %krange[jj])
            axs[ii, 0].set_ylabel('Clusters: %d' %numclusters[ii])
            fig.suptitle('Label embeddings viewed with '+embedding, fontsize='large')


def plot_TSNE(U,
              i2l,
              numclusters,
              seed=0):
    """ Make one plot of the TSNE embedding of the label points given
    in matrix U.

    """
    clusters, cl = kmeans(U,i2l,numclusters)
    model = TSNE(n_components=2, random_state=seed)
    tsne = model.fit_transform(U)
    clustercolors = [i/(U.shape[0]+0.0) for i in clusters]
    cm = plt.cm.get_cmap('RdYlBu')
    avg_norm = np.mean([norm(tsne[i]) for i in range(tsne.shape[0]) ])
    tsne = tsne/avg_norm
    plt.scatter(tsne[:,0],tsne[:,1], c=clustercolors, s=35,cmap=cm)
    for i in range(tsne.shape[0]):
        plt.text(tsne[i,0], tsne[i,1], i2l[i])
    #plt.show()


def distance_matrix(U, l2i, i2l):
    """ Get distances between every pair of points; each point is a row of U.

    """
    n = U.shape[0]
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            distances[i,j] = np.linalg.norm(U[i]-U[j])
            distances[i,i] = 1e16

    closest = [np.argmin(distances[i]) for i in range(n)]

    closest_labels = {}
    for i in l2i.keys():
        closest_labels[i] = i2l[closest[l2i[i]]]
    return distances, closest_labels


