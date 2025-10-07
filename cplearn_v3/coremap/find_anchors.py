
from ..utils.densify import densify_v0
from ..utils.gen_utils import bipartite_NN

from ..corespect.layer_extraction import find_intersection_layers

import leidenalg
import numpy as np

import time


from joblib import Parallel, delayed

from sklearn.mixture import GaussianMixture

def determine_gmm_anchors(X_core,min_k=1,max_k=10,reg_covar=1e-6, random_state=0):
    """
    X_core: (n_core, 2) UMAP coords for core points
    Returns:
      labels: (n_core,) hard cluster labels in [0..k-1]
      centers: (k, 2) component means (anchors)
      model: fitted GaussianMixture
    """

    bic_list=[]

    n = len(X_core)
    if n == 0:
        raise ValueError("No core points provided.")
    max_k = max(min_k, min(max_k, n))  # cap by data size

    # Try k=1..max_k, pick best by BIC
    best = None
    k_pos=-1
    best_bic = np.inf
    for k in range(min_k, max_k + 1):
        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",      # 'diag' is even faster; 'full' is fine in 2D
            init_params="kmeans",
            n_init=3,
            max_iter=200,
            reg_covar=reg_covar,
            random_state=random_state,
        ).fit(X_core)
        bic = gm.bic(X_core)
        bic_list.append(bic)

        if bic < best_bic:
            best_bic, best = bic, gm
            k_pos=k

        #This is for fast mode.
        if bic > best_bic:
            break

    labels  = best.predict(X_core)
    centers = best.means_          # use as skeleton anchor points
    return centers,labels


def choose_stopping_res(G,core_nodes,thr=0.95):

    n0=len(core_nodes)

    n1=n0
    t=1
    while n1>thr*n0:
        layer_candidate=find_intersection_layers(G,core_layer=core_nodes,rem_nodes=core_nodes,res=t)
        n1=len(layer_candidate)
        t+=0.25
        print(t,n1,n0)

    return t-0.25


#The resolution can be made parameter free in the visualization step.
def find_anchors(core_obj):


    G=core_obj.G
    adj_list=np.array(core_obj.knn_list).astype(int)
    X=core_obj.X


    layers_=core_obj.layers_
    #densify=core_obj.densify
    densify=False

    core_nodes=layers_[0]

    G_core = G.induced_subgraph(core_nodes)

    if densify:
        print("densifying...")
        G_core = densify_v0(adj_list, G, G_core, core_nodes)

    #Obtain the suitable resolution.
    res=choose_stopping_res(G,core_nodes,thr=0.95)

    partition = leidenalg.find_partition(
        G_core,
        leidenalg.RBConfigurationVertexPartition,  # modularity-based partition
        resolution_parameter=res,  # <-- control resolution here
        seed=np.random.randint(int(1e9))
    )



    filtered_partition=[]
    for cluster in partition:
        if len(cluster)>5:
            filtered_partition.append(cluster)

    partition=filtered_partition

    print("Total number of clusters=",len(partition))
    print([len(cluster) for cluster in partition])

    t1=time.time()



    anchors_dict = Parallel(n_jobs=-1)(
        delayed(determine_gmm_anchors)(X[np.array(list(cluster)).astype(int)]) for cluster in partition)

    t2=time.time()


    print(f"GMM time={t2-t1:.3f} seconds (corrected)")

    anchor_list = []
    anchors = []
    t=0
    for i,cluster in enumerate(partition):

        ng_list=[[] for _ in range(len(anchors_dict[i][0]))]
        temp_labels=anchors_dict[i][1]
        n_1=len(temp_labels)
        for node in range(n_1):
            ng_list[temp_labels[node]].append(G_core.vs["id"][cluster[node]])




        for ng in ng_list:
            anchor_list.append(ng)
            anchors.append(np.mean(X[np.array(ng).astype(int)],axis=0))

        #Send the points connected to each center in the GMM-on-Leiden setup.


    anchor_distances = []
    for i in range(len(anchor_list)):
        vec=[]
        for j in anchor_list[i]:
            vec.append(np.linalg.norm(anchors[i]-X[j]))

        anchor_distances.append(vec)

    anchored_list_sorted=[]
    anchored_dist_sorted=[]
    for idxs, dists in zip(anchor_list, anchor_distances):
        # zip indexes with distances, sort by distance, unzip back
        pairs = sorted(zip(idxs, dists), key=lambda x: x[1])
        idx_sorted, dist_sorted = zip(*pairs)

        anchored_list_sorted.append(list(idx_sorted))
        anchored_dist_sorted.append(list(dist_sorted))

    return anchored_list_sorted,anchored_dist_sorted





