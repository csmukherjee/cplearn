import numpy as np
import umap
from .coremap_utils import process_data
from .coremap_utils import k_means_for_skeleton
from .anchored_map import anchored_map

from collections import Counter

import numpy as np
from sklearn.mixture import GaussianMixture

def gmm_on_viz_core(X_core, k_max=10, min_k=1, reg_covar=1e-6, random_state=0):
    """
    X_core: (n_core, 2) UMAP coords for core points
    Returns:
      labels: (n_core,) hard cluster labels in [0..k-1]
      centers: (k, 2) component means (anchors)
      model: fitted GaussianMixture
    """
    n = len(X_core)
    if n == 0:
        raise ValueError("No core points provided.")
    k_max = max(min_k, min(k_max, n))  # cap by data size

    # Try k=1..k_max, pick best by BIC
    best = None
    best_bic = np.inf
    for k in range(min_k, k_max + 1):
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
        if bic < best_bic:
            best_bic, best = bic, gm

    labels  = best.predict(X_core)
    centers = best.means_          # use as skeleton anchor points
    return labels, centers



def get_anchors(X,points,label_round,method='centroid',anchor_method='variable_k_gmm'):

    if method=='centroid':
        n_sub=len(points)
        out_k=len(set(list(label_round)))
        centroids=np.zeros((out_k,2))
        class_counts=np.zeros(out_k)

        X_core=X[points]

        anchors=[]

        for ell in range(n_sub):
            centroids[label_round[ell]]+=X_core[ell]
            class_counts[label_round[ell]]+=1

        for j in range(out_k):
            centroids[j]/=class_counts[j]
            anchors.append([centroids,j])


        return anchors

    elif method=='skeleton':
        n_sub=len(points)
        out_k=len(set(list(label_round)))
        clusters=[[] for _ in range(out_k)]
        class_counts = np.zeros(out_k)
        X_core = X[points]

        for ell in range(n_sub):
            clusters[label_round[ell]].append(points[ell])


        anchors = []


        t=0
        for cluster in clusters:
            n_part=len(cluster)

            if n_part<20:
                centroids=np.zeros((1,2))
                for point in cluster:
                    centroids[0]+=point

                centroids[0]/=n_part

            else:
                if anchor_method=='fixed_k':

                    centroids=[]
                    if n_part<40:
                        k_here=2


                    else:
                        k_here=min(int(np.floor(np.log2(len(cluster))+1)),6)


                    centroids_here, labels_km = k_means_for_skeleton(X[np.array(cluster).astype(int)], true_k=k_here)
                    co=Counter(labels_km)
                    for ell in co:
                        if co[ell]>n_part/(2*k_here):
                            centroids.append(centroids_here[ell])

                elif anchor_method=='variable_k_gmm':
                    centroids = []
                    if n_part < 40:
                        k_here = 2


                    else:
                        k_here = min(int(np.floor(np.log2(len(cluster)) + 1)), 6)

                    labels_km, centroids_here=gmm_on_viz_core(X[np.array(cluster).astype(int)], k_max=k_here)
                    co=Counter(labels_km)
                    for ell in co:
                        if co[ell]>n_part/(4*k_here):
                            centroids.append(centroids_here[ell])



                else:
                    raise ValueError("Unknown anchor method.")

            for ell in centroids:
                anchors.append([ell,t]) #Remember this encoding for next step.

            t+=1


        #print("Total clusters and total anchors selected=",out_k, len(anchors))

        return anchors


    else:
        raise ValueError("Method not recognized.")



def make_to_dict(embedding,points):

    out_dict={}

    for i,ell in enumerate(points):
        out_dict[ell]=embedding[i]

    return out_dict


def cmap_multi_layer(G,init_embedding,all_points,cores,anchors,all_points_labels):


    embedding=anchored_map(G, all_points, cores, all_points_labels, anchors, init_embedding.copy(), q=15)

    return embedding


def cmap_cores(G,init_embedding,cores,core_labels):

    anchors=get_anchors(init_embedding,cores,core_labels,method='skeleton')

    core_embedding=anchored_map(G, cores, cores, core_labels, anchors, init_embedding.copy(), q=15)

    return core_embedding,anchors



def anchored_cmap(coremap_obj,X,round_info,final_labels,mode='three_steps',q=15,global_init_embedding=None):

    total_embedding_dict={}

    total_embedding_list=[]
    label_list=[]

    #Get the layers and set the cores
    G,layers,labels=process_data(X,round_info,final_labels,q,mode=mode)
    cores=np.array(layers[0]).astype(int)
    core_labels=np.array(labels[0]).astype(int)

    #Initial umap embedding for anchoring
    if global_init_embedding is None:
        init_embedding=umap.UMAP(n_neighbors=q,init='spectral' ).fit_transform(X)
    else:
        init_embedding=global_init_embedding


    #Now perform the anchored map for the cores.
    curr_embedding,anchors=cmap_cores(G,init_embedding, cores, core_labels)
    total_embedding_dict["cores"]=make_to_dict(curr_embedding,cores)

    total_embedding_list.append(curr_embedding)
    label_list.append(core_labels)

    layer_nodes=layers[0]
    layer_labels=labels[0]

    for round_num in range(1,len(layers)):

        layer_nodes.extend(layers[round_num])
        layer_labels.extend(labels[round_num])

        layer_labels_to_send=np.array(layer_labels).astype(int)

        #print(len(layer_nodes),len(layer_labels),round_num)

        curr_embedding=cmap_multi_layer(G,init_embedding,layer_nodes,cores=cores,anchors=anchors,all_points_labels=layer_labels_to_send)

        total_embedding_dict["Layer"+str(int(round_num))]=make_to_dict(curr_embedding,layer_nodes)

        total_embedding_list.append(curr_embedding)
        label_list.append(layer_labels.copy())




    return total_embedding_dict