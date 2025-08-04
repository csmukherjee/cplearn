from utils import ranking, propagate
from corespect.clustering import cluster_core
import numpy as np


def corespect(X, true_k=None, q=40, r=20, ng_num=20, ranking_algo='FlowRank', layer_ratio=None, scores=None, cluster_algo='k_means',propagate_algo='CDNN',cav='dist',
                send_meta_data=False):


    num_step = len(layer_ratio) - 1
    n = X.shape[0]

    #If scores are not provided, rank using ranking_algo
    if scores is None:
        if hasattr(ranking, ranking_algo):
            func = getattr(ranking, ranking_algo)
            if callable(func):
                scores=func(X, q, r)
            else:
                raise TypeError(f"{ranking_algo} is not callable")

        else:
            raise KeyError(f"{ranking_algo} is not a valid ranking algorithm")
        print(f"Obtained vertex ranking using {ranking_algo}")

    else:
        print("Ranking with passed scores")

    sorted_points = np.array(sorted(scores, key=scores.get, reverse=True)).astype(int)


    if layer_ratio is not None:
        top_frac = layer_ratio[0] #Need to make this parameter free at some point.

    else:
        raise KeyError("The C-P partitions cannot be None")

    core_nodes = sorted_points[0:int(top_frac * n)]


    #Now, we cluster the core_nodes using cluster_algo
    if hasattr(cluster_core, cluster_algo):
        func = getattr(cluster_core, cluster_algo)
        if callable(func):
            core_labels,cluster_assignment_vectors = func(X, core_nodes=core_nodes,true_k=true_k,cav=cav,ng_num=ng_num)
        else:
            raise TypeError(f"{cluster_algo} is not callable")

    else:
        raise KeyError(f"{cluster_algo} is not a valid clustering algorithm")


    # Start generating final labels
    final_labels = -1 * np.ones(n)
    final_labels[core_nodes] = core_labels
    final_labels = final_labels.astype(int)

    print("Number of clusters found in the core:", len(set(core_labels)))

    #Now we label the rest (or some of the rest) of the points using the propagate_algo algorithm
    if hasattr(propagate, propagate_algo):
        func = getattr(propagate, propagate_algo)
        if callable(func):
            final_labels= func(X,sorted_points,core_nodes,final_labels,layer_ratio,cluster_assignment_vectors,ng_num)
        else:
            raise TypeError(f"{cluster_algo} is not callable")

    else:
        raise KeyError(f"{propagate_algo} is not a valid propagation algorithm")



    return final_labels
