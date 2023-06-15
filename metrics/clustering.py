import numpy as np
import sklearn
import sklearn.manifold
import sklearn.cluster
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import _supervised
from scipy.optimize import linear_sum_assignment
import scipy.sparse
import torch

def feature_detection(A, label):
    """ Computes proportion of l_1 norm
    of each row of A that is given to connections outside of cluster
    """
    n = A.shape[0]
    err = 0
    # TODO: get rid of for loop
    for i in range(n):
        err += np.abs(A[i,label==label[i]]).sum()/np.abs(A[i,:]).sum()
    err /= n
    err = 1-err
    return err

def nmi(label, pred_label):
    return normalized_mutual_info_score(label, pred_label)

def percent_wrong_edge(A, label):
    row, col = A.nonzero()
    matches = label[row] != label[col]
    if isinstance(matches, torch.Tensor):
        matches = matches.float()
    return matches.mean()*100

def clustering_accuracy(label, pred_label):
    """ from https://github.com/ChongYou/subspace-clustering
    """
    label, pred_label = _supervised.check_clusterings(label, pred_label)
    value = _supervised.contingency_matrix(label, pred_label)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(label)

def sparsity(A, zero_cutoff=1e-8):
    """ Average number of nonzeros per row
    """
    return np.sum(np.abs(A) > zero_cutoff)/A.shape[0]
def basic_metrics(A, label, verbose=True):
    nnz = sparsity(A)
    fd_error = feature_detection(A, label)
    components = scipy.sparse.csgraph.connected_components(A, return_labels=False)
    wrong_edge = percent_wrong_edge(A, label)
    if verbose:
        print(f"NNZ/ row: {nnz:.2f}   ||| Feat detect: {fd_error:.5f} ")
        print(f"Num comp: {components}       ||| Pct wrong edges: {wrong_edge:.2f}")
    return nnz, fd_error, components, wrong_edge

def spectral_clustering_metrics(A, nclass, label, verbose=True, n_init=10, normalize_embed=True, solver_type='lm', extra_dim=0, tol=0):
    """ n_init is number of separate runs of kmeans to average over
    computes average accuracy and nmi
    """
    lap = scipy.sparse.csgraph.laplacian(A, normed=True)
    nnz, fd_error, components, wrong_edge = basic_metrics(A, label, verbose=False)
    if components > nclass:
        print('---Oversegmented graph, setting higher eigensolver tolerance (unstable results)---')
        # oversegmented, need higher tolerance
        tol = 1e-4
        
    if solver_type=='shift_invert':
        vals, embedding = scipy.sparse.linalg.eigsh(lap, k=nclass+extra_dim, sigma=1e-6, which='LM', tol=tol)
    elif solver_type=='la':
        vals, embedding = scipy.sparse.linalg.eigsh(-lap, k=nclass+extra_dim,
                                    sigma=None,  which='LA', tol=tol)
    elif solver_type=='lm':
        vals, embedding = scipy.sparse.linalg.eigsh(
            2*scipy.sparse.identity(lap.shape[0])-lap,
            k=nclass+extra_dim, sigma=None,  which='LM', tol=tol)
    else:
        raise ValueError('invalid solver')
    
    if normalize_embed:
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    cluster_model = sklearn.cluster.KMeans(n_clusters=nclass, n_init=n_init)
    acc_lst = []
    nmi_lst = []
    pred_lst = []
    for _ in range(n_init):
        cluster_model.fit(embedding)
        pred_label = cluster_model.labels_
        acc = clustering_accuracy(label, pred_label)
        nmi_score = nmi(label, pred_label)
        acc_lst.append(acc)
        nmi_lst.append(nmi_score)
        pred_lst.append(pred_label)
        
    #conn_lst = connectivity_lst(A, label)
    
    if verbose:
        print(f'Acc mean: {np.mean(acc_lst):.3f}   ||| stdev: {np.std(acc_lst):.4f}')
    if components > nclass:
        # do not record unstable results for oversegmented case
        acc_lst = [0]
    
    return acc_lst, nmi_lst, fd_error, nnz, pred_lst