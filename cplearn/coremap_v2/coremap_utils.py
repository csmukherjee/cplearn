from ..utils.processing import extract_layers_labels

from . import coremap_graph_utils as cgutils

import numpy as np



import networkx as nx

def _fix_graphical_cores(X,layers,labels,q=15):
    n=X.shape[0]
    G = cgutils.get_weighted_graph(X, q)

    core_nodes=layers[0]

    hmap=np.zeros(n)
    for ell in core_nodes:
        hmap[ell]=1

    H=nx.DiGraph()

    for u,v in G.edges():
        if hmap[u]==1 and hmap[v]==1:
            H.add_edge(u,v,weight=G.edges[u,v]['weight'])


    filtered_cores=np.array(list(H.nodes())).astype(int)

    new_layers0=[]
    new_labels0=[]
    for idx in range(len(layers[0])):

        ell=layers[0][idx]

        if ell in filtered_cores:
            new_layers0.append(ell)
            new_labels0.append(labels[0][idx])

        else:
            layers[1].append(ell)
            labels[1].append(labels[0][idx])


    layers[0]=new_layers0
    labels[0]=new_labels0

    assert len(layers[0]) == len(labels[0]), "Length mismatch after fixing cores."


    return G,layers,labels





def process_data(X,round_info,final_labels,q=15,mode='three_steps'):

        layers, labels=extract_layers_labels(round_info,final_labels,mode=mode)
        G = cgutils.get_weighted_graph(X, q)

        #G,layers,labels=_fix_graphical_cores(X,layers,labels,q)

        return G,layers,labels


from sklearn.cluster import KMeans
def k_means_for_skeleton(X,true_k,choose_min_obj=True):

    if choose_min_obj:
        min_obj_val = float('inf')

        for rounds in range(5):

            kmeans = KMeans(n_clusters=true_k, n_init=1, max_iter=1000)
            kmeans.fit(X)

            centroids = kmeans.cluster_centers_
            obj_val = kmeans.inertia_
            labels_km = kmeans.labels_

            if rounds == 0 or obj_val < min_obj_val:
                min_obj_val = obj_val
                best_centroids = centroids
                best_labels_km = labels_km

        centroids = best_centroids
        labels_km = best_labels_km

    else:
        kmeans = KMeans(n_clusters=true_k)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        labels_km = kmeans.labels_


    return centroids, labels_km