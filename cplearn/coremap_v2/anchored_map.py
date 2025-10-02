
from . import coremap_graph_utils as cutils
from .coremap_graph_utils import varying_heat_kernel
from ..utils import graph_utils
from ..utils import densify

import networkx as nx
import numpy as np

from numba import njit, prange

def _make_epochs_per_sample(weights, n_epochs):
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return w
    result = np.full(w.shape[0], -1.0, dtype=np.float64)  # sentinel: never sample
    wmax = float(w.max())
    if wmax <= 0.0:
        return result
    n_samples = n_epochs * (w / wmax)
    mask = n_samples > 0.0
    result[mask] = float(n_epochs) / n_samples[mask]
    return result




@njit
def clip(val):
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


@njit
def clip_th(val,th=4.0):
    if val > th:
        return th
    elif val < -th:
        return -th
    else:
        return val

@njit  # keep single-threaded to avoid races on 'coordinates'
def _optimize_layout_euclidean_single_epoch(
    coordinates,            # (n, dim)
    weights,                # (|E|,)
    edge_from,              # (|E|,)
    edge_to,                # (|E|,)
    epochs_per_sample,      # (|E|,)
    epoch_of_next_sample,   # (|E|,)
    epochs_per_negative_sample,        # (|E|,)
    epoch_of_next_negative_sample,     # (|E|,)
    n,                      # current epoch index
    alpha,                  # learning rate
    a, b,
    repulsion_ratio
):
    n_vertices = coordinates.shape[0]
    dim = coordinates.shape[1]

    for i in range(epochs_per_sample.shape[0]):

        # ---- Positive (attractive) sample ----
        if epoch_of_next_sample[i] <= n:
            j = edge_from[i]
            k = edge_to[i]

            current = coordinates[j]
            other   = coordinates[k]

            # squared distance
            dist2 = 0.0
            for d in range(dim):
                diff = current[d] - other[d]
                dist2 += diff * diff

            if dist2 > 0.0:
                grad_coeff = -2.0 * a * b * (dist2 ** (b - 1.0))
                grad_coeff /= (a * dist2 + 1.0)
            else:
                grad_coeff = 0.0


            # Update BOTH endpoints for attraction
            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                delta  = grad_d * alpha
                current[d] += delta
                other[d]   -= delta  # symmetric update

            epoch_of_next_sample[i] += epochs_per_sample[i]

            # ---- Negative (repulsive) samples ----
            # Count how many negative updates are due at epoch n
            n_neg = 0
            if epochs_per_negative_sample[i] < 1e308:  # not inf
                if epoch_of_next_negative_sample[i] <= n:
                    # include current epoch; floor division equivalent
                    n_neg = int((n - epoch_of_next_negative_sample[i]) /
                                epochs_per_negative_sample[i]) + 1
                    if n_neg < 0:
                        n_neg = 0  # safety

            for _ in range(n_neg):
                # simple RNG; reproducibility not guaranteed under njit
                kk = np.random.randint(n_vertices)
                if kk == j:
                    continue
                other = coordinates[kk]

                dist2 = 0.0
                for d in range(dim):
                    diff = current[d] - other[d]
                    dist2 += diff * diff

                if dist2 > 0.0:
                    grad_coeff = (2.0 * b) / ((0.001 + dist2) * (a * (dist2 ** b) + 1.0))
                else:
                    grad_coeff = 0.0

                if grad_coeff > 0.0:
                    for d in range(dim):
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                        current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += n_neg * epochs_per_negative_sample[i]



def _stabilize_layout_with_anchor_single_epoch(
    coordinates,            # (n, dim),            # (n, dim)
    weights,                # (|E|,)
    edge_from,              # (|E|,)
    coordinates_to,                # (|E|,)
    epochs_per_sample,      # (|E|,)
    epoch_of_next_sample,   # (|E|,)
    n,                      # current epoch index
    alpha,                  # learning rate
    a, b
):
    dim = coordinates.shape[1]

    for i in range(epochs_per_sample.shape[0]):

        # ---- Positive (attractive) sample ----
        if epoch_of_next_sample[i] <= n:
            j = edge_from[i]
            current = coordinates[j]

            other=coordinates_to[i]


            # squared distance
            dist2 = 0.0
            for d in range(dim):
                diff = current[d] - other[d]
                dist2 += diff * diff

            if dist2 > 0.0:
                grad_coeff = -2.0 * a * b * (dist2 ** (b - 1.0))
                grad_coeff /= (a * dist2 + 1.0)
            else:
                grad_coeff = 0.0


            # Update BOTH endpoints for attraction
            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                delta  = grad_d * alpha
                current[d] += delta
                #other[d]   -= delta  # symmetric update

            epoch_of_next_sample[i] += epochs_per_sample[i]



def _stabilize_layout_with_anchor_single_epoch_old(coordinates, anchors, anchored_graph, label_round, points, init_embedding, n, alpha, a, b):

    dim=coordinates.shape[1]

    for idx, anchor_idx in anchored_graph.edges():

        current=coordinates[idx]
        other=anchors[anchor_idx]


        force=1


        #curr_dist = np.linalg.norm(coordinates[idx] - init_embedding[points[idx]]) #This is based on overall movement so far

        curr_dist=np.linalg.norm(coordinates[idx]-anchors[anchor_idx]) #This is based on distance to anchor


        if curr_dist > 0.0:
            grad_coeff = -2.0 * a * b * (curr_dist ** (b - 1.0))*force
            grad_coeff /= (a * (curr_dist ** b) + 1.0)
        else:
            grad_coeff = 0.0

        for d in range(dim):
            grad_d = clip(grad_coeff * (current[d] - other[d]))
            delta = grad_d * alpha
            current[d] += delta





def _prepare_anchored_edge_list_with_skeleton(G, points, anchor_points, labels_round, anchors, embedding, q=15):
    n=G.number_of_nodes()
    hmap=np.zeros(n)
    for ell in points:
        hmap[ell]=1

    H=nx.DiGraph()

    for ell in points:
        H.add_node(ell)

    for u,v in G.edges():
        if hmap[u]==1 and hmap[v]==1:
            H.add_edge(u,v,weight=G.edges[u,v]['weight'])


    filtered_cores=np.array(list(H.nodes())).astype(int)

    assert len(filtered_cores)==len(points) , "This should not happen. Core was already filtered"

    n_sub=len(points)

    H=densify.Densify_v0(G, H.copy(), points, q)



    #Convert to contiguous indices
    anchor_idx_ch=np.zeros(n)
    for ell in anchor_points:
        anchor_idx_ch[ell]=1

    #Linear mapping
    hmap_f=np.zeros(n)
    anchor_idx_f=np.zeros(n_sub)
    t=0
    for ell in points:
        hmap_f[ell]=t

        if anchor_idx_ch[ell]==1:
            anchor_idx_f[t]=1

        t+=1



    #Creating a graph that is re-indexed to 0...n_sub-1
    H_f=nx.Graph()
    for u,v in H.edges():
        uu = int(hmap_f[u])
        vv = int(hmap_f[v])
        wt=H.edges[u,v]['weight']
        if uu < 0 or vv < 0:
            continue
        if H_f.has_edge(uu, vv):
            H_f[uu][vv]["weight"] += wt
        else:
            H_f.add_edge(uu, vv, weight=wt)

    #Convert to umap friendly structure
    m = H_f.number_of_nodes()
    edge_from, edge_to, weights_final = [], [], []
    for u, v, data in H_f.edges(data=True):
        edge_from.append(u); edge_to.append(v); weights_final.append(data["weight"])

    weights_final = np.asarray(weights_final, dtype=np.float64)


    #New anchoring.
    out_k=len(set(labels_round))
    anchor_sets=[[] for ell in range(out_k)]
    for ell in anchors:
        anchor_sets[ell[1]].append(ell[0])


    anchor_points_map={}
    anchor_points_rev_map={}
    t=0
    for ell in range(n_sub):
        if anchor_idx_f[ell]==1:
            anchor_points_map[ell]=t
            anchor_points_rev_map[t]=ell
            t+=1


    sk_distances=[[] for _ in range(len(anchor_points))]
    sk_lists=[[] for _ in range(len(anchor_points))]
    for ell in range(n_sub):

        if anchor_idx_f[ell]==1:
            anchor_idx=anchor_points_map[ell]
            label_idx=labels_round[ell]
            for ell_p in anchor_sets[label_idx]:
                sk_distances[anchor_idx].append(np.linalg.norm(embedding[anchor_points[anchor_idx]]-ell_p))
                sk_lists[anchor_idx].append(ell_p)

    # for xx in range(len(sk_lists)):
    #     print(len(sk_lists[xx]),len(sk_distances[xx]),end=', ')

    print('\n')

    assert len(sk_lists) == len(sk_distances), "Row counts must match"
    sorted_lists = []
    sorted_dists = []
    for nbrs, dists in zip(sk_lists, sk_distances):
        assert len(nbrs) == len(dists), "Row lengths must match"
        if not nbrs:  # empty row
            sorted_lists.append([])
            sorted_dists.append([])
            continue
        pairs = sorted(zip(dists, nbrs), key=lambda x: x[0])
        sd, sn = zip(*pairs)  # tuples
        sorted_dists.append(list(sd))
        sorted_lists.append(list(sn))

    sk_lists=sorted_lists
    sk_distances=sorted_dists
    P_vec=varying_heat_kernel(sk_distances)



    anchor_edges_from=[]
    anchor_coordinates_to=[]
    anchor_weights_final=[]
    for i in range(len(sk_lists)):
        ell=anchor_points_rev_map[i]
        for j in range(len(sk_lists[i])):
            anchor_edges_from.append(ell)
            anchor_coordinates_to.append(sk_lists[i][j])
            anchor_weights_final.append(P_vec[i][j])




    return edge_from, edge_to, weights_final, anchor_edges_from, anchor_coordinates_to, anchor_weights_final


def _prepare_anchored_edge_list(G, points, anchor_points, labels_round, anchors, embedding, q=15):

    n=G.number_of_nodes()
    hmap=np.zeros(n)
    for ell in points:
        hmap[ell]=1

    H=nx.DiGraph()

    for ell in points:
        H.add_node(ell)

    for u,v in G.edges():
        if hmap[u]==1 and hmap[v]==1:
            H.add_edge(u,v,weight=G.edges[u,v]['weight'])


    filtered_cores=np.array(list(H.nodes())).astype(int)

    assert len(filtered_cores)==len(points) , "This should not happen. Core was already filtered"

    n_sub=len(points)

    H=densify.Densify_v0(G, H.copy(), points, q)

    #Convert to contiguous indices
    anchor_idx=np.zeros(n)
    for ell in anchor_points:
        anchor_idx[ell]=1


    #Linear mapping
    hmap_f=np.zeros(n)
    anchor_idx_f=np.zeros(n_sub)
    t=0
    for ell in points:
        hmap_f[ell]=t

        if anchor_idx[ell]==1:
            anchor_idx_f[t]=1

        t+=1



    #Creating a graph that is re-indexed to 0...n_sub-1
    H_f=nx.Graph()
    for u,v in H.edges():
        uu = int(hmap_f[u])
        vv = int(hmap_f[v])
        wt=H.edges[u,v]['weight']
        if uu < 0 or vv < 0:
            continue
        if H_f.has_edge(uu, vv):
            H_f[uu][vv]["weight"] += wt
        else:
            H_f.add_edge(uu, vv, weight=wt)

    #Convert to umap friendly structure
    m = H_f.number_of_nodes()
    edge_from, edge_to, weights_final = [], [], []
    for u, v, data in H_f.edges(data=True):
        edge_from.append(u); edge_to.append(v); weights_final.append(data["weight"])

    weights_final = np.asarray(weights_final, dtype=np.float64)




    #anchored graph.
    anchored_graph = nx.DiGraph()
    for idx in range(n_sub):

        if anchor_idx_f[idx]==1:
            cen=anchors[labels_round[idx]]
            dist=np.linalg.norm(embedding[filtered_cores[idx]]-cen)
            anchored_graph.add_edge(idx,labels_round[idx],og_dist=dist)



    return edge_from, edge_to, weights_final,anchored_graph




def anchored_map(G, points, anchor_points, label_round, anchors, embedding, q=15):

#    edge_from, edge_to, weights_final, anchored_graph=_prepare_anchored_edge_list(G, points, anchor_points, label_round, anchors, embedding, q=15)

    edge_from, edge_to, weights_final, anchor_edges_from, anchor_coordinates_to, anchor_weights_final=_prepare_anchored_edge_list_with_skeleton(G, points, anchor_points, label_round, anchors, embedding, q=15)


    coordinates=embedding[points]
    coordinates = np.ascontiguousarray(coordinates, dtype=np.float64)


    #Setup values for the normal edges.
    edge_from = np.asarray(edge_from, dtype=np.int64)
    edge_to   = np.asarray(edge_to,   dtype=np.int64)
    weights   = np.asarray(weights_final,   dtype=np.float64)


    #Setting umap parameters
    total_epochs=400
    repulsion_ratio=5.0
    initial_alpha=1.0
    a=1.579
    b=0.895



    epochs_per_sample = _make_epochs_per_sample(weights, total_epochs)

    # schedule: never-sampled edges => +inf, not negative numbers
    epoch_of_next_sample = np.where(epochs_per_sample > 0.0,
                                    epochs_per_sample.copy(),
                                    np.inf)

    epochs_per_negative_sample = np.where(epochs_per_sample > 0.0,
                                          epochs_per_sample / repulsion_ratio,
                                          np.inf)
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()



    #setup the values for anchored_edges
    anchor_edge_from = np.asarray(anchor_edges_from, dtype=np.int64)
    anchor_edge_to   = np.asarray(anchor_coordinates_to,   dtype=np.int64)
    anchor_weights   = np.asarray(anchor_weights_final,   dtype=np.float64)

    print(np.shape(anchor_edge_from), np.shape(anchor_edge_to), np.shape(anchor_weights))


    anchor_epochs_per_sample = _make_epochs_per_sample(anchor_weights, total_epochs)

    # schedule: never-sampled edges => +inf, not negative numbers
    anchor_epoch_of_next_sample = np.where(anchor_epochs_per_sample > 0.0,
                                    anchor_epochs_per_sample.copy(),
                                    np.inf)




    alpha = float(initial_alpha)

    for n in range(total_epochs):
        _optimize_layout_euclidean_single_epoch(
            coordinates,
            weights, edge_from, edge_to,
            epochs_per_sample,
            epoch_of_next_sample,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            n, alpha, a, b, repulsion_ratio
        )
        alpha = initial_alpha * (1.0 - (n + 1.0) / float(total_epochs))
        if alpha < 0.0:
            alpha = 0.0


        _stabilize_layout_with_anchor_single_epoch(
            coordinates,
            anchor_weights, anchor_edge_from, anchor_edge_to,
            anchor_epochs_per_sample,
            anchor_epoch_of_next_sample,
            n, alpha, a, b
        )



    return coordinates




