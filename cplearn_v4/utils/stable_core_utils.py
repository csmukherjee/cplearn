
import numpy as np

from joblib import Parallel, delayed

from ..utils.gen_utils import ensure_list_of_lists

from numba import njit, prange

import leidenalg



def make_subgraph(adj_list, core_nodes):
    import igraph as ig

    n = len(adj_list)
    edges = [(u, v) for u in range(n) for v in adj_list[u] if u != v]

    G = ig.Graph(edges=edges, directed=True)
    G.vs["id"] = list(range(n))  # Store original node ids

    G_core = G.induced_subgraph(core_nodes)

    return G,G_core







def choose_stopping_res(adj_list,core_nodes,thr=0.95,starting_resolution=1):

    n0=len(core_nodes)

    n1=n0
    t=starting_resolution
    while n1>thr*n0 and t<5:
        layer_candidate=find_intersection_layers(adj_list,core_nodes,ng_num=20,rem_nodes=np.array(core_nodes).astype(int),resolution=starting_resolution)
        n1 = len(layer_candidate)
        print(t,n1,n0)
        t += 0.25


    return t-0.5


def cluster_subset(adj_list, core_nodes, resolution, **kwargs):

    """
    Cluster the core subgraph using the Leiden algorithm (directed version).

    Parameters
    ----------
    adj_list : list[list[int]]
        Adjacency list of the full directed graph.
    core_nodes : list[int]
        Indices of nodes forming the core subset.
    resolution : float
        Resolution parameter for Leiden clustering.
    **kwargs :
        Additional keyword arguments passed to leidenalg.find_partition.

    Returns
    -------
    labels : np.ndarray of shape (n,)
        Cluster labels for all nodes, with -1 for non-core nodes.
    """


    n=len(adj_list)
    G,G_core= make_subgraph(adj_list,core_nodes)

    partition = leidenalg.find_partition(
        G_core,
        leidenalg.RBConfigurationVertexPartition,  # modularity-based partition
        resolution_parameter=resolution,  # <-- control resolution here
        seed=np.random.randint(int(1e9)), **kwargs
    )

    id_map = np.array(G_core.vs["id"], dtype=int)

    labels = -1 * np.ones(n, dtype=int)
    labels[id_map] = partition.membership

    return labels



@njit(parallel=True)
def build_edges_and_out_degree_numba(adj_list):
    n = len(adj_list)

    # Count total edges
    total_edges = 0
    for u in range(n):
        total_edges += len(adj_list[u])

    edge_list = np.empty((total_edges, 2), dtype=np.int64)
    out_degree = np.empty(n, dtype=np.int64)

    offset = 0
    for u in prange(n):  # parallel outer loop
        nbrs = adj_list[u]
        out_degree[u] = len(nbrs)
        for v in nbrs:
            edge_list[offset, 0] = u
            edge_list[offset, 1] = v
            offset += 1

    return edge_list, out_degree




def majority_single_iteration(edge_list,deg_list,n, cnum, init_labels, rem_nodes_mask,eps=0):

    votes = np.zeros((n, cnum))
    added_nodes = []

    for u, v in edge_list:

        c_idx = int(init_labels[v])
        if rem_nodes_mask[u] == 1 and c_idx != -1:
            votes[u, c_idx] += 1

    for u in range(n):

        if rem_nodes_mask[u] == 1:
            if max(votes[u]) > (0.5 + eps) * deg_list[u]:
                init_labels[u] = np.argmax(votes[u])
                rem_nodes_mask[u] = 0
                added_nodes.append(u)

    return init_labels, rem_nodes_mask, added_nodes



def recursive_majority(adj_list_n, core_nodes, rem_nodes, resolution=1):

    adj_list=ensure_list_of_lists(adj_list_n)

    n=len(adj_list)
    tolerance = int(0.005 * n)
    max_iter=500 #SafeGuard to avoid infinite loops

    init_labels= cluster_subset(adj_list, core_nodes, resolution=resolution)

    cnum= len(set(init_labels)) - (1 if -1 in init_labels else 0)
    edge_list, deg_list=build_edges_and_out_degree_numba(adj_list)

    rem_nodes_mask=np.zeros(n).astype(int)
    rem_nodes_mask[rem_nodes]=1

    layer=[]

    round_num=0
    while True:
        n0 = np.sum(init_labels != -1)
        proxy_labels, rem_nodes_mask,added_nodes = majority_single_iteration(edge_list,deg_list,n, cnum, init_labels, rem_nodes_mask)



        for ell in added_nodes:
            layer.append(ell)

        n1 = np.sum(proxy_labels != -1)

        if n1 - n0 < tolerance:
            return layer


        if n0 > 0.95 * n:
            return layer

        if round_num>max_iter:
            return layer

        round_num += 1






def find_intersection_layers(adj_list,core_nodes,rem_nodes,ng_num,resolution):
    n=len(adj_list)



    if rem_nodes is None:
        rem_nodes = np.array([i for i in range(n) if i not in core_nodes])

    np.random.shuffle(rem_nodes) #Should not effect reproducibility


    layer_list = Parallel(n_jobs=10)(
        delayed(recursive_majority)(adj_list, core_nodes, rem_nodes,resolution) for _ in range(10))

    layer_candidate = set(layer_list[0])
    for i in range(1, 10):
        layer_candidate = layer_candidate.intersection(set(layer_list[i]))

    return layer_candidate