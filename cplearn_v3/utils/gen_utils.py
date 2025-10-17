import numpy as np

import hnswlib
import igraph as ig
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
        for j_loc in knn_list_local[i_loc]:
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


def get_igraph_from_knn_list(knn_list,n):
    e_list = get_edge_list(knn_list, n)
    G = ig.Graph(directed=True)
    G.add_vertices(n)
    G.add_edges(e_list)
    G.vs["id"] = list(range(n))

    return G

from numba import types
from numba.typed import List
import numpy as np

def get_induced_knn(knn_list, knn_dist, S, n):
    """
    Filter knn_list/knn_dist so each row keeps only neighbors in S.
    Returns numba.typed.List[List[int64]], numba.typed.List[List[float64]]
    ready for @njit use.
    """
    in_subset = -1*np.ones(n, dtype=np.uint64)
    for local_id, global_id in enumerate(S):
        in_subset[global_id] = local_id


    # Ensure list-of-lists form
    if isinstance(knn_list, np.ndarray):
        knn_list = knn_list.tolist()
        knn_dist = knn_dist.tolist()

    # Top-level typed lists
    new_knn_list = List.empty_list(types.ListType(types.int64))
    new_knn_dist = List.empty_list(types.ListType(types.float64))


    for i in range(n):
        if in_subset[i]>=0:
            inner_nodes = List.empty_list(types.int64)
            inner_dists = List.empty_list(types.float64)

            nbrs=knn_list[i]
            dists=knn_dist[i]
            for j in range(len(nbrs)):

                v=int(nbrs[j])
                if in_subset[v]>=0:
                    inner_nodes.append(int(in_subset[v]))
                    inner_dists.append(float(dists[j]))


            new_knn_list.append(inner_nodes)
            new_knn_dist.append(inner_dists)

    return new_knn_list, new_knn_dist




