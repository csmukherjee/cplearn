
from sklearn.cluster import KMeans
import numpy as np

def k_means(X,core_nodes,true_k,choose_min_obj=True,cav='dist',ng_num=15):

    X_core = X[core_nodes]

    if choose_min_obj:
        min_obj_val = float('inf')

        for rounds in range(20):

            kmeans = KMeans(n_clusters=true_k, n_init=1, max_iter=1000)
            kmeans.fit(X_core)

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
        kmeans.fit(X_core)
        centroids = kmeans.cluster_centers_
        labels_km = kmeans.labels_


    cluster_assignment_vectors=[]

    if cav=='dist':
        for i in range(len(core_nodes)):
            vec = []
            for j in range(true_k):
                vec.append(np.linalg.norm(X_core[i] - centroids[j]))
            cluster_assignment_vectors.append(np.array(vec).astype('float64'))


    if cav=='ind':
        for i in range(len(core_nodes)):
            vec = []
            for j in range(true_k):
                if labels_km[i] == j:
                    vec.append(-1)
                else:
                    vec.append(0)
            cluster_assignment_vectors.append(np.array(vec).astype('float64'))

    #cluster_assignment_vectors = np.array(cluster_assignment_vectors)

    print("Clustered core using k-means with cav:", cav)

    return labels_km,cluster_assignment_vectors


def spectral_clustering(X,core_nodes,true_k,choose_min_obj=True,cav='ind',ng_num=15):
    from sklearn.manifold import SpectralEmbedding

    X_core = X[core_nodes]

    SE = SpectralEmbedding(n_components=true_k, affinity='nearest_neighbors', n_neighbors=15)
    X_core = SE.fit_transform(X_core)

    if choose_min_obj:
        min_obj_val = float('inf')

        for rounds in range(20):

            kmeans = KMeans(n_clusters=true_k, n_init=1, max_iter=1000)
            kmeans.fit(X_core)

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
        kmeans.fit(X_core)
        centroids = kmeans.cluster_centers_
        labels_km = kmeans.labels_


    cluster_assignment_vectors=[]

    if cav=='dist':
        for i in range(len(core_nodes)):
            vec = []
            for j in range(true_k):
                vec.append(np.linalg.norm(X_core[i] - centroids[j]))
            cluster_assignment_vectors.append(np.array(vec).astype('float64'))


    if cav=='ind':
        for i in range(len(core_nodes)):
            vec = []
            for j in range(true_k):
                if labels_km[i] == j:
                    vec.append(-1)
                else:
                    vec.append(0)
            cluster_assignment_vectors.append(np.array(vec).astype('float64'))

    print("Clustered core using spectral-clustering with cav:", cav)

    return labels_km,cluster_assignment_vectors


import louvain_setup
from collections import deque

import networkx as nx



M=200
ef_construction=200
ef=200

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


def louvain(X,core_nodes,true_k,choose_min_obj=True,cav='ind',ng_num=15,resolution=1.0):


    n=X.shape[0]

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

    print(len(core_nodes),Gp.number_of_nodes())

    total_partition=louvain_setup.louvain_partitions(Gp, weight="weight", resolution=resolution)
    partition_ = deque(total_partition, maxlen=1).pop()
    label_map=louvain_setup.partition_to_label(partition_)

    core_labels=[]
    for ell in  core_nodes:
        core_labels.append(label_map[ell])


    our_k = len(set(core_labels)) - (1 if -1 in core_labels else 0)

    cluster_assignment_vectors=[]

    if cav=='dist':
        raise KeyError("CAV 'dist' not supported for louvain clustering")

    if cav=='ind':
        for i in range(len(core_nodes)):
            vec = []
            for j in range(our_k):
                if core_labels[i] == j:
                    vec.append(-1)
                else:
                    vec.append(0)
            cluster_assignment_vectors.append(np.array(vec).astype('float64'))

    print("Clustered core using louvain with cav:", cav)



    return core_labels,cluster_assignment_vectors

def hdbscan(X,core_nodes,true_k,choose_min_obj=True,cav='ind',min_cluster_size=10,min_sample=10,ng_num=15):


    import hdbscan

    X_core = X[core_nodes]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_sample,
        metric='l2',
        alpha=1.1,
        prediction_data=True
    ).fit(X_core)

    from hdbscan import all_points_membership_vectors

    soft_probs = all_points_membership_vectors(clusterer)
    core_labels = np.argmax(soft_probs, axis=1)


    our_k= len(set(core_labels)) - (1 if -1 in core_labels else 0)

    cluster_assignment_vectors=[]

    if cav=='dist':
        raise KeyError("CAV 'dist' not supported for HDBSCAN")

    if cav=='ind':
        for i in range(len(core_nodes)):
            vec = []
            for j in range(our_k):
                if core_labels[i] == j:
                    vec.append(-1)
                else:
                    vec.append(0)
            cluster_assignment_vectors.append(np.array(vec).astype('float64'))


    core_labels_final=np.ones(len(core_nodes)) * -1

    label_map={}
    t=0
    for ell in set(core_labels):
        label_map[ell]=t
        t+=1

    for i in range(len(core_nodes)):
        core_labels_final[i] = label_map[core_labels[i]]


    return core_labels_final,cluster_assignment_vectors

