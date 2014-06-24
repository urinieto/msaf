import numpy as np
from sets import Set

def pairwise( annotations ):
    """Computes the pairwise set of distances from the annotations

    @param annotations array: Annotations of the labels.
    @return P set: Pairwise annotations to compute the F-measure of similarity.
    """
    P = Set()
    N = len(annotations)
    for i in range(N):
        l1 = annotations[i]
        for j in range(i+1,N):
            l2 = annotations[j]
            #print i, j, l1, l2
            if l1 == l2:
                P.add((i,j))

    return P

def beat_sync_labels(labels, bounds, beats):
    """Syncrhonizes the labels at a beat level based on the boundaries
     (used for evaluation purposes). New labels have the same length than
     len(beats).

    @param labels np.array: Labels at a beat level.
    @param bounds np.array: Boundaries (in seconds).
    @param beats np.array: Beat times (in seconds).
    @return sync_labels: Beat-synced labels.
    """
    bound_beats = []
    bound_idx = 1
    sync_labels = []
    for beat in beats:
        if bound_idx >= len(bounds):
            sync_labels += [labels[-1]]*(len(beats) - len(sync_labels))
            break
        try:
            sync_labels.append(labels[bound_idx-1])
        except:
            sync_labels.append(labels[-1])
        if beat >= bounds[bound_idx]:
            bound_idx += 1
    return np.asarray(sync_labels)

def eval_segmentation_entropy(estframelabs, gtframelabs):
    """
    Evaluates the similarity of labels using the entropy method.

    @param estframelabs dict: Estimated labels.
    @param gtframelabs dict: Ground truth labels.

    @return So float: Over segmentation entropy value.
    @return Su float: Under segmentation entropy value.
    """

    #A = np.array([2,2,2,1,1,1,2,2,2,1,1,1])
    #E = np.array([1,2,1,2,1,2,1,2,1,2,1,2])
    #gtframelabs = A
    #estframelabs = E

    gtlabels = np.unique(gtframelabs)
    estlabels = np.unique(estframelabs)

    N = len(estframelabs)
    Na = len(gtlabels)
    Ne = len(estlabels)

    nij = np.zeros((Na, Ne))
    nia = np.zeros((Na))
    nje = np.zeros((Ne))
    for i in range(Na):
        curra = 1.0 * (gtframelabs == gtlabels[i]) + np.spacing(1)
        nia[i] = np.sum(curra)
        for j in range(Ne):
            curre = 1.0 * (estframelabs == estlabels[j])
            nij[i,j] = np.dot(curra, np.transpose(curre)) + np.spacing(1)
    for j in range(Ne):
        curre = 1.0 * (estframelabs == estlabels[j]) + np.spacing(1)
        nje[j] = np.sum(curre)

    norm = np.sum(np.sum(nij))
    pij = nij / norm
    pia = nia / norm
    pje = nje / norm
    pijae = nij / np.tile(nje, (Na, 1))
    pjiea = np.transpose(nij) / np.tile(nia, (Ne,1))

    HEA = - np.sum(pia * np.sum(pjiea * np.log2(pjiea), axis=0))
    HAE = - np.sum(pje * np.sum(pijae * np.log2(pijae), axis=0))

    So = 1 - HEA / np.log2(Ne)
    Su = 1 - HAE / np.log2(Na)

    if np.isnan(So) or np.isinf(So):
        So = 0
    if np.isnan(Su) or np.isinf(Su):
        Su = 0

    return So, Su


def eval_similarity(e_labels, a_labels, bounds, beats, ver=False):
    """
    Evaluates the similarity using the pair-wise frame clustering method.

    @param e_labels np.array: Array with estimated labels.
    @param a_labels np.array: Array with annotated labels.
    @param bounds np.array: Array with bounds (in seconds).
    @param beats np.array: Array with beats (in seconds).

    @return F float: F-measure.
    @return P float: Precision.
    @return R float: Recall.
    """

    # "Beat-synchronize" labels for pair-wise frame eval
    e_sync_labels = beat_sync_labels(e_labels, bounds, beats)
    a_sync_labels = beat_sync_labels(a_labels, bounds, beats)

    # Pairwise Set from estimated results
    Pe = pairwise( e_sync_labels )

    # Pairwise Set from human anotations
    Pa = pairwise( a_sync_labels )

    # Precision
    P = len(Pe & Pa) / float(len(Pe))

    # Recall
    R = len(Pe & Pa) / float(len(Pa))

    # F-measure
    if R + P == 0: F = 0
    else: F = 2. * R * P / (R + P)

    if ver:
        print 'Similarity Eval: F: %.2f\tP: %.2f\tR: %.2f\t'% (F*100, 
                                                               P*100, R*100)

    return F, P, R

def eval_boundaries(e_bounds, a_bounds, sec=3, ver=False):
    """Evaluates the boundaries.

    @param e_bounds np.array: Estimated boundaries.
    @param a_bounds np.array: Annotated boundaries.
    @param sec float: Threshold for evaluating boundaries (in seconds).
    @param ver bool: Verbose mode.

    @return F float: F-measure.
    @return P float: Precision.
    @return R float: Recall.
    """
    # Compute positive hits
    positives = 0
    """
    for e_bound in e_bounds:
        e_bound_min = e_bound - sec
        e_bound_max = e_bound + sec
        for a_bound in a_bounds:
            if a_bound > e_bound_min and a_bound < e_bound_max:
                positives += 1
                break
                """
    for a_bound in a_bounds:
        a_bound_min = a_bound - sec
        a_bound_max = a_bound + sec
        for e_bound in e_bounds:
            if e_bound > a_bound_min and e_bound < a_bound_max:
                positives += 1
                break

    # Precision
    if len(e_bounds) != 0: 
        P = positives / float(len(e_bounds))
    else: 
        P = 0
    if P > 1: P = 1 # This may only happen in extreme cases

    # Recall
    if len(a_bounds) != 0: 
        R = positives / float(len(a_bounds))
    else: 
        R = 0

    # F-Measure
    if R + P == 0: F = 0
    else: F = 2. * R * P / (R + P)

    if ver:
        print "Boundaries Eval: F:%.2f  P:%.2f  R:%.2f" % (F*100, P*100, R*100)

    return F, P, R
