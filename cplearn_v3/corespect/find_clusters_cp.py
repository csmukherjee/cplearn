#First finish this.
from collections import Counter
from ..utils.stable_core_extraction import *



import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"


from typing import Optional



import numpy as np
import scipy.sparse as sp



def cluster_core(corespect_obj,res_t=None):
    G=corespect_obj.G
    layers=corespect_obj.layers_
    core_nodes=layers[0]

    if res_t is None:
        res_t=choose_stopping_res(G,core_nodes)#,starting_res=corespect_obj.resolution)

    #res_t=choose_stopping_res(G,core_nodes)

    labels=cluster_subset(G,core_nodes,res_t)
    return labels




def igraph_to_sparse(G, weight_attr="weight",direction='out'):
    """
    Convert an igraph.Graph to a scipy.sparse.csr_matrix adjacency.

    Parameters
    ----------
    G : igraph.Graph
        The input graph (directed or undirected).
    weight_attr : str, optional
        Name of the edge attribute to use as weight.
        If not found or None, all edges get weight 1.
    make_symmetric : bool, optional
        If True, symmetrizes the adjacency (useful for undirected graphs).

    direction: 'in', 'out', or 'sym' 'in'=> in-edges only (This should later be changed to remove P->C edges only.
    Returns
    -------
    A : scipy.sparse.csr_matrix
        Sparse adjacency matrix.
    """
    n = G.vcount()
    edges = np.array(G.get_edgelist(), dtype=np.int64)

    # Edge weights (or 1 if unweighted)
    if weight_attr is not None and weight_attr in G.edge_attributes():
        weights = np.array(G.es[weight_attr], dtype=float)
    else:
        weights = np.ones(len(edges), dtype=float)

    A = sp.csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n))

    # Make symmetric if undirected
    if direction == 'in':
        A=A.T
    elif direction == 'sym':

        A = A + A.T
        A.setdiag(0)
        A.eliminate_zeros()

    return A



import numpy as np
import scipy.sparse as sp

def diffusion_layers(P: sp.csr_matrix,
                     F0: np.ndarray,
                     core_mask: np.ndarray,
                     classes: np.ndarray,
                     max_iter: int = 30,
                     tol: float = 1e-4,
                     diffusion_mode='default'):
    """
    Diffusion-based layers via temporal absorption.

    Steps:
      1) Run diffusion normally (no masking).
      2) Record argmax(F_t[v]) at each iteration.
      3) Final label = argmax(F_final[v]).
      4) layer[v] = first t where argmax(F_t[v]) == argmax(F_final[v]).

    Parameters
    ----------
    P : csr_matrix (n x n)
        Row-normalized transition matrix (D^{-1}A).
    F0 : (n x K) float
        One-hot initialization for cores, zeros elsewhere.
    core_mask : (n,) bool
        True for cores (absorbing).
    classes : (K,) int
        Class IDs corresponding to columns of F.
    max_iter : int
        Maximum diffusion iterations.
    tol : float
        Global convergence tolerance on max(|F_{t+1} - F_t|).

    diffusion_mode: Two versions: Default and momentum

    Returns
    -------
    hard_labels : (n,) int
        Final hard class assignments (original class IDs; -1 if undecided).
    F_final : (n x K) float
        Final diffusion scores.
    layer_idx : (n,) int
        First iteration where node's predicted label matched its final label.
        (cores have layer 0)
    """
    n, K = F0.shape
    F = F0.copy()
    F_core_fixed = F0[core_mask].copy()

    # --- record argmax trajectory to save memory ---
    argmax_history = np.empty((max_iter + 1, n), dtype=np.int32)
    argmax_history[0] = F.argmax(axis=1)

    # --- diffusion loop ---
    for t in range(1, max_iter + 1):

        if diffusion_mode=='default':
            F_new = P @ F

        F_new[core_mask] = F_core_fixed
        diff = np.max(np.abs(F_new - F))
        F = F_new
        argmax_history[t] = F.argmax(axis=1)
        if diff < tol:
            # truncate history to actual number of iterations
            argmax_history = argmax_history[: t + 1]
            break
    else:
        # no early break; use full length
        t = max_iter

    F_final = F.copy()
    argmax_final = argmax_history[-1]

    # --- compute layer for each node ---
    layer_idx = np.full(n, -1, dtype=int)
    for v in range(n):
        if core_mask[v]:
            layer_idx[v] = 0
            continue
        final_label = argmax_final[v]
        matches = np.flatnonzero(argmax_history[:, v] == final_label)
        # first occurrence of the final label in the trajectory
        layer_idx[v] = int(matches[0]) if matches.size > 0 else t

    # --- final hard labels ---
    row_max = F_final.max(axis=1)
    hard_labels = np.full(n, -1, dtype=np.int64)
    nz = row_max > 0
    hard_labels[nz] = classes[argmax_final[nz]]

    return hard_labels, F_final, layer_idx

def label_spread_absorbing_fast(
    A_csr: sp.csr_matrix,
    y: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-4,
    dtype=np.float32,
    num_threads: Optional[int] = None,
    diffusion_mode='default'
):
    """
    Fast label spreading with absorbing cores + diffusion-defined layers.

    Returns
    -------
    hard_labels : (n,) int
    F : (n x K) float
    layer_idx : (n,) int
    stabilized_iter : (n,) int
    """
    # threading hints (OpenBLAS/OMP)
    if num_threads is not None and num_threads > 0:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))

    if not sp.isspmatrix_csr(A_csr):
        A_csr = A_csr.tocsr(copy=False)

    n = A_csr.shape[0]
    y = np.asarray(y, dtype=np.int64)
    if y.shape[0] != n:
        raise ValueError("y must have length equal to A_csr.shape[0]")

    core_mask = (y >= 0)
    if not np.any(core_mask):
        # Nothing to propagate
        return y.copy(), np.zeros((n, 0), dtype=dtype), np.zeros(n, dtype=int), np.zeros(n, dtype=int)

    # classes and mapping (stable order)
    classes = np.unique(y[core_mask])
    K = int(classes.shape[0])
    lab2idx = {int(c): i for i, c in enumerate(classes)}

    # build row-stochastic P = D^{-1} A (in-place, O(nnz))
    P = A_csr.copy().astype(dtype, copy=False)
    indptr = P.indptr
    data = P.data
    d = np.asarray(P.sum(axis=1)).ravel().astype(dtype)
    invdeg = 1.0 / np.maximum(d, np.finfo(dtype).tiny)
    counts = np.diff(indptr)
    data *= np.repeat(invdeg, counts)

    # F0 one-hot on cores
    F0 = np.zeros((n, K), dtype=dtype)
    labeled_idx = np.where(core_mask)[0]
    F0[labeled_idx, [lab2idx[int(lbl)] for lbl in y[labeled_idx]]] = 1.0

    hard, F, layer_idx, stabilized_iter = diffusion_layers(
        P, F0, core_mask, classes=classes,
        max_iter=max_iter, tol=tol, delta_thr=1e-4,diffusion_mode=diffusion_mode
    )

    return hard, F, layer_idx

def majority(corespect_obj,find_cluster_params=None):

    #gmm_on_core=find_cluster_params.get('gmm_on_core',False)
    if find_cluster_params is not None:
        res_core=find_cluster_params.get('res',1)

    res_core=None

    direction=find_cluster_params.get('direction','out')
    diffusion_mode=find_cluster_params.get('diffusion_mode','default')

    layers=corespect_obj.layers_

    final_labels=cluster_core(corespect_obj,res_core)
    rem_nodes=[]
    for node in layers[1]:
        if final_labels[node] == -1:
            rem_nodes.append(node)

    G=corespect_obj.G
    cnum=len(set(final_labels))-1

    print("Remaining nodes=",len(rem_nodes))

    curr_layers=layers[0]+layers[1]

    layer_num=1


    Adj = igraph_to_sparse(G,direction=direction)

    _, final_labels, remaining_nodes = partitioned_majority_stage1(G, cnum, final_labels,
                                                                 np.array(rem_nodes).astype(int), 0)

    print(Counter(final_labels))

    final_labels, prob_mat, layer_idx=label_spread_absorbing_fast(Adj, final_labels,diffusion_mode=diffusion_mode)

    print(np.min(layer_idx),np.max(layer_idx),len(set(layer_idx)))


    new_layers=[[] for _ in range(len(set(layer_idx)))]
    for i in range(len(final_labels)):
        new_layers[layer_idx[i]].append(i)

    corespect_obj.new_layers=new_layers








    # while layer_num<len(layers):
    #     print("First try Remaining nodes=", remaining_nodes.shape, np.count_nonzero(final_labels == -1))
    #
    #     if remaining_nodes.shape[0]>0:
    #         _, final_labels, remaining_nodes,_ = partitioned_majority_stage2(G, cnum, final_labels,
    #                                                                         remaining_nodes, 0)
    #         print("second try Remaining nodes=", remaining_nodes.shape, np.count_nonzero(final_labels == -1))
    #
    #     layer_num += 1
    #
    #     if layer_num<len(layers):
    #         rem_nodes = []
    #         curr_layers.extend(layers[layer_num])
    #         for node in curr_layers:
    #             if final_labels[node] == -1:
    #                 rem_nodes.append(node)
    #



    return final_labels

def CDNN(X,layers,find_cluster_params=None):

    labels=None

    return labels
