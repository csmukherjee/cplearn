from ..utils import louvain_setup

import numpy as np

import hnswlib
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


from ..utils import densify
import networkx as nx
from collections import deque

def louvain(X,core_nodes,cluster_algo_params):



    allowed_params = ['ng_num', 'resolution']
    for key in cluster_algo_params.keys():
        if key not in allowed_params:
            raise ValueError(f"Unwanted parameter found: {key}")

    ng_num = cluster_algo_params.get('ng_num', 15)
    resolution = cluster_algo_params.get('resolution', 1.0)

    n=X.shape[0]

    print("Clustering core with louvain, ng_num=",ng_num," resolution=",resolution)

    #densify

    G= nx.DiGraph()
    knn_list, _ = get_kNN(X, q=ng_num)
    for i in range(n):
        for j in knn_list[i, :]:
            if i != j:
                G.add_edge(i, j, weight=1)

    hmap=np.zeros(n)
    for ell in core_nodes:
        hmap[ell] = 1

    Gp= nx.DiGraph()
    for ell in core_nodes:
        Gp.add_node(ell)

    for u,v in G.edges():
        if hmap[u] == 1 and hmap[v] == 1:
            Gp.add_edge(u, v, weight=G.edges[u, v]['weight'])

#    Gp=densify.Densify_v0(G, Gp, core_nodes, ng_num)
#    print(len(core_nodes),Gp.number_of_nodes())


    total_partition=louvain_setup.louvain_partitions(Gp, weight="weight", resolution=resolution)
    partition_ = deque(total_partition, maxlen=1).pop()
    label_map=louvain_setup.partition_to_label(partition_)

    core_labels=[]
    for ell in  core_nodes:
        core_labels.append(label_map[ell])



    #Reorder labels
    hmap={}
    t=0
    for ell in set(core_labels):
        hmap[ell]=t
        t+=1

    new_core_labels=[]
    for ell in range(len(core_labels)):
        new_core_labels.append(hmap[core_labels[ell]])

    core_labels=new_core_labels.copy()

    print("Clustered core using louvain")


    return core_labels