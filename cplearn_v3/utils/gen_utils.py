import numpy as np

import hnswlib
from numba import njit

M=200
ef_construction=200
ef=200

def get_kNN(X, q=15):
    """
    Generate a k-nearest neighbors graph from the input data.
    :param X: Input data (numpy array).
    :param q: Number of nearest neighbors.
    :return: k-nearest neighbors list and distances.
    """
    n = X.shape[0]
    dim = X.shape[1]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=n, ef_construction=200, M=64)
    p.add_items(X)
    p.set_ef(2*q)

    labels, dists = p.knn_query(X, k=q+1)
    knn_list = labels[:, 1:]
    knn_dists = dists[:, 1:]

    return knn_list, knn_dists


def bipartite_NN(X,Y, q=15):
    """
    Generate a k-nearest neighbors graph from the input data.
    :param X: Input data (numpy array).
    :param q: Number of nearest neighbors.
    :return: k-nearest neighbors list and distances.
    """
    n = X.shape[0]
    dim = X.shape[1]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=n, ef_construction=200, M=64)
    p.add_items(X)
    p.set_ef(2*q)

    labels, dists = p.knn_query(Y, k=q+1)
    knn_list = labels[:, 1:]
    knn_dists = dists[:, 1:]

    return knn_list, knn_dists



@njit
def get_edge_list(knn_list_local,n):
    edge_list=[]
    for i_loc in range(n):
        for j_loc in knn_list_local[i_loc, :]:
            if i_loc != j_loc:
                edge_list.append((i_loc,j_loc))

    return edge_list



@njit
def get_induced_edge_list(edge_list,subset,n):

    hmap=-1*np.ones(n)
    for ell in subset:
        hmap[ell]=1

    new_edge_list=[]

    for u,v in edge_list:
        if hmap[u]==1 and hmap[v]==1:
            new_edge_list.append((u,v))

    return new_edge_list