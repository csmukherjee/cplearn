import numpy as np
import scipy
from flatbuffers.packer import float64


from ..utils import graph_utils
from ..utils import densify

def varying_heat_kernel(distances,return_mode='default'):
    n_points = len(distances)
    sigmas = np.zeros(n_points)
    target = np.zeros(n_points)
    rhos = np.zeros(n_points)

    P_vec = [[] for _ in range(n_points)]

    # Step 1: Compute rho_i (local connectivity)
    for i in range(n_points):

        rhos[i] = distances[i][0] if len(distances[i]) > 0 else 0  # Minimum nonzero distance

        if rhos[i] != min(distances[i]):
            raise KeyError(f"0-th index is not closest, {rhos[i]:.3f} {min(distances[i]):.3f}")

    # Step 2: Solve for sigma_i using binary search
    def find_sigma(i, target_i):

        lo, hi = 1e-8, 100000.0  # Search range for sigma
        for _ in range(64):  # Binary search
            sigma = (lo + hi) / 2.0
            weights = np.exp(-(distances[i] - rhos[i]) / sigma)
            if np.sum(weights) > target_i:
                hi = sigma
            else:
                lo = sigma
        return (lo + hi) / 2.0

    for i in range(n_points):
        if len(distances[i]) > 0:
            c1 = len(distances[i])

            if c1 == 1:
                target[i] = 1

            elif c1 <= 5:
                target[i] = 1 + c1 / 2

            else:
                target[i] = np.log2(c1)

            sigmas[i] = find_sigma(i, target[i])

    # Step 3: Compute edge weights
    for i in range(n_points):
        for d in distances[i]:
            weight = np.exp(-(d - rhos[i]) / sigmas[i])

            P_vec[i].append(weight)


    if return_mode=='anchor':
        return P_vec, sigmas, rhos, target

    return P_vec


def make_bidirectional(G):

    for u, v in G.edges():
        if G.has_edge(v, u):
            w1 = G.edges[u, v]['weight']
            w2 = G.edges[v, u]['weight']
            df = w1 + w2 - (w1 * w2)
            G.edges[u, v]['weight'] = df
            G.edges[v, u]['weight'] = df

    return G

import networkx as nx



def get_weighted_graph(X,q=15,bidirectional=True):
    n=X.shape[0]
    knn_list,knn_distances=graph_utils.get_kNN(X,q)
    P_vec, sigmas, rhos, target=varying_heat_kernel(knn_distances,return_mode='anchor')
    P_vec=np.array(P_vec).astype(np.float64)

    print("test2")
    print(np.shape(P_vec),np.shape(knn_list),np.shape(knn_distances))
    print(np.min(P_vec),np.max(P_vec))
    for ell in range(10,100,10):
        print(np.percentile(P_vec,ell),end=' ')

    print('\n')

    G=nx.DiGraph()
    for i in range(n):
        for j in range(len(knn_list[i])):
            v=knn_list[i][j]
            if i!=v:
                G.add_edge(i,v,weight=P_vec[i][j])

    if bidirectional:
        G=make_bidirectional(G)

    for u in G.nodes():
        G.nodes[u]['sigma']=sigmas[u]

    return G

def weight_densified_graph(X,H):

    nodes=list(H.nodes())
    hmap={}
    t=0
    for ell in range(len(nodes)):
        hmap[nodes[ell]]=t
        t=t+1

    n_sub=len(nodes)


    knn_list=[[] for _ in range(n_sub)]
    knn_distances=[[] for _ in range(n_sub)]
    for u,v in H.edges():
        knn_list[hmap[u]].append(v)
        knn_distances[hmap[u]].append(np.linalg.norm(X[u]-X[v]))

    n_sub=len(knn_list)

    assert len(knn_list) == len(knn_distances), "Row counts must match"
    sorted_lists = []
    sorted_dists = []
    for nbrs, dists in zip(knn_list, knn_distances):
        assert len(nbrs) == len(dists), "Row lengths must match"
        if not nbrs:  # empty row
            sorted_lists.append([])
            sorted_dists.append([])
            continue
        pairs = sorted(zip(dists, nbrs), key=lambda x: x[0])
        sd, sn = zip(*pairs)  # tuples
        sorted_dists.append(list(sd))
        sorted_lists.append(list(sn))

    knn_list=sorted_lists
    knn_distances=sorted_dists

    P_vec=varying_heat_kernel(knn_distances)
    for i in range(n_sub):
        for j in range(len(knn_list[i])):
            v=knn_list[i][j]
            if i!=v:
                H.edges[nodes[i],v]['weight']=P_vec[i][j]

    return H


def get_sparse_coo_graph(G,q=15):

    n=G.number_of_nodes()
    rows=[]
    cols=[]
    vals=[]
    for u,v in G.edges():
        rows.append(u)
        cols.append(v)
        vals.append(G.edges[u,v]['weight'])

    graph = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n,n))


    return graph

def get_induced_subgraph(X,G,core_nodes,q=15,densification=True):

    n=X.shape[0]
    hmap=np.zeros(n)
    for ell in core_nodes:
        hmap[ell]=1


    H=nx.DiGraph()
    for u,v in G.edges():
        if hmap[u]==1 and hmap[v]==1:
            H.add_edge(u,v,weight=G.edges[u,v]['weight'])


    refined_core_nodes=list(H.nodes())

    if densification:
        H=densify.Densify_v0(G,H.copy(),refined_core_nodes,q)



    H=weight_densified_graph(X,H)
    #H=make_bidirectional(H)



    return H








