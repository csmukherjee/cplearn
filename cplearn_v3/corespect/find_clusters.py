# -*- coding: utf-8 -*-
"""
Dataclass-based rewrite with:
- Clean configuration objects
- Correct diffusion operator usage (P or S), not Laplacian multiplication
- Safe argmax (no false matches for all-zero rows)
- Temporal-absorption layer assignment (first t where argmax matches final)
- Optional momentum update
- Contiguous layer remapping with core layer fixed to 0
- Directional graph conversion (in/out/sym)

External dependencies (left as in original code):
- choose_stopping_res, cluster_subset, partitioned_majority_stage1
  from your utils module(s).
"""
from __future__ import annotations

from ..utils.stable_core_extraction import *

from dataclasses import dataclass, asdict, replace, field
from typing import Optional, Tuple, List

import os
import numpy as np
import scipy.sparse as sp

# External functions expected to be available in your package:
# from ..utils.stable_core_extraction import *
# (kept commented to avoid import errors here; restore in your project)
# from ..utils.stable_core_extraction import choose_stopping_res, cluster_subset, partitioned_majority_stage1


# ------------------------------
# Configuration dataclasses
# ------------------------------

@dataclass
class DiffusionConfig:
    # Operator & iteration
    norm_mode: str = "normalized"          # 'normalized' -> S = D^{-1/2} A D^{-1/2}, 'random_walk' -> P = D^{-1} A
    diffusion_mode: str = "momentum"        # 'default' (pure operator) or 'momentum'
    max_iter: int = 30
    tol: float = 1e-4
    dtype: np.dtype = np.float32

    # Momentum step size (only if diffusion_mode='momentum')
    alpha: float = 0.9

    # Threading hints (OpenBLAS/OMP); set when running the algorithm
    num_threads: Optional[int] = None

    # Layering options
    contiguous_layers: bool = True         # remap discrete layers to contiguous 0..L
    start_t_for_noncore: int = 1           # ignore matches at t=0 for non-cores (prevents layer 0 leakage)

    # Optional confidence gating for layering (None -> disabled)
    # If set (e.g., 0.6), we must know F_t[v, final_label] per t, which requires
    # storing snapshots. To keep memory bounded, this is disabled by default.
    cluster_threshold: Optional[float] = None
    store_full_snapshots: bool = False     # only needed if using cluster_threshold


@dataclass
class MajorityConfig:
    direction: str = "out"                 # 'out' | 'in' | 'sym'
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    cluster_resolution: Optional[float] = None


# ------------------------------
# Graph utilities
# ------------------------------

def igraph_to_sparse(G, weight_attr: str = "weight", direction: str = "out") -> sp.csr_matrix:
    """
    Convert an igraph.Graph to a scipy.sparse.csr_matrix adjacency.
    direction: 'out' -> edges i->j, 'in' -> transpose, 'sym' -> symmetrize (no double counting).
    """
    import numpy as _np
    import scipy.sparse as _sp

    n = G.vcount()
    edges = _np.array(G.get_edgelist(), dtype=_np.int64)

    # Edge weights (or 1 if unweighted)
    if weight_attr is not None and weight_attr in G.edge_attributes():
        weights = _np.array(G.es[weight_attr], dtype=float)
    else:
        weights = _np.ones(len(edges), dtype=float)

    A = _sp.csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n))

    if direction == "in":
        A = A.T
    elif direction == "sym":
        # Prefer max to avoid doubling weights if the graph is already undirected
        A = A.maximum(A.T)
        A.setdiag(0)
        A.eliminate_zeros()

    return A


def build_diffusion_operator(A_csr: sp.csr_matrix,
                             mode: str = "normalized",
                             dtype=np.float64) -> sp.csr_matrix:
    """
    Returns the diffusion operator, not the Laplacian:
      - mode='random_walk' -> P = D^{-1} A
      - mode='normalized'  -> S = D^{-1/2} A D^{-1/2}
    O(nnz) in-place scaling of CSR data.
    """
    if not sp.isspmatrix_csr(A_csr):
        A_csr = A_csr.tocsr(copy=False)
    A_csr = A_csr.astype(dtype, copy=False)

    n = A_csr.shape[0]
    d = np.asarray(A_csr.sum(axis=1)).ravel().astype(dtype)
    d_safe = np.maximum(d, np.finfo(dtype).tiny)

    B = A_csr.copy()
    indptr, indices, data = B.indptr, B.indices, B.data

    if mode == "random_walk":
        invdeg = 1.0 / d_safe
        counts = np.diff(indptr)
        data *= np.repeat(invdeg, counts)     # in-place P = D^{-1}A
        B.eliminate_zeros()
        return B

    if mode == "normalized":
        inv_sqrt = 1.0 / np.sqrt(d_safe)
        for i in range(n):
            s, e = indptr[i], indptr[i + 1]
            if e > s:
                data[s:e] *= inv_sqrt[i] * inv_sqrt[indices[s:e]]  # S = D^{-1/2} A D^{-1/2}
        B.eliminate_zeros()
        return B

    raise ValueError("mode must be 'random_walk' or 'normalized'")


def build_laplacian(A_csr: sp.csr_matrix,
                    mode: str = "normalized",
                    dtype=np.float64) -> sp.csr_matrix:
    """
    For completeness: Laplacians if you need them elsewhere.
      - 'random_walk': L_rw  = I - D^{-1} A
      - 'normalized' : L_sym = I - D^{-1/2} A D^{-1/2}
    """
    B = build_diffusion_operator(A_csr, mode=mode, dtype=dtype)
    L = sp.eye(B.shape[0], dtype=dtype, format="csr") - B
    L.eliminate_zeros()
    return L


# ------------------------------
# Diffusion & layers
# ------------------------------

def _apply_thread_env(num_threads: Optional[int]) -> None:
    if num_threads is not None and num_threads > 0:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))


def diffusion_layers(A_csr: sp.csr_matrix,
                     F0: np.ndarray,
                     core_mask: np.ndarray,
                     classes: np.ndarray,
                     cfg: DiffusionConfig
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diffusion-based layers via temporal absorption:

      1) Run diffusion with operator (P or S), keeping cores absorbing.
      2) Record argmax(F_t[v]) at each iteration (safe argmax: -1 if row is all-zero).
      3) Final label = argmax(F_final[v]).
      4) layer[v] = first t >= cfg.start_t_for_noncore where argmax(F_t[v]) == argmax(F_final[v]).
         (cores forced to layer 0)
      5) Optionally remap layers to contiguous integers.

    Returns:
      hard_labels (n,), F_final (n x K), layer_idx (n,)
    """
    _apply_thread_env(cfg.num_threads)

    n, K = F0.shape
    F = F0.astype(cfg.dtype, copy=True)
    F_core_fixed = F0[core_mask].astype(cfg.dtype, copy=True)

    # diffusion operator
    P = build_diffusion_operator(A_csr, mode=cfg.norm_mode, dtype=cfg.dtype)

    # record argmax trajectory
    T_max = cfg.max_iter
    argmax_history = np.empty((T_max + 1, n), dtype=np.int32)

    # t=0 safe argmax
    row_max0 = F.max(axis=1)
    argmax0 = F.argmax(axis=1)
    argmax0[row_max0 <= 1e-12] = -1
    argmax_history[0] = argmax0

    # Optional snapshots (only if confidence gating requested)
    F_snaps: List[np.ndarray] = []
    if cfg.cluster_threshold is not None and cfg.store_full_snapshots:
        F_snaps.append(F.copy())

    # diffusion loop
    t_last = 0
    for t in range(1, T_max + 1):
        if cfg.diffusion_mode == "default":
            F_new = P @ F
        elif cfg.diffusion_mode == "momentum":
            alpha = cfg.alpha
            F_new = (1 - alpha) * F + alpha * (P @ F)
        else:
            raise KeyError("Unknown diffusion_mode: %r" % cfg.diffusion_mode)

        # enforce absorbing cores
        F_new[core_mask] = F_core_fixed

        diff = np.max(np.abs(F_new - F))
        F = F_new
        t_last = t

        # inline safe argmax
        row_max = F.max(axis=1)
        argmax = F.argmax(axis=1)
        argmax[row_max <= 1e-12] = -1
        argmax_history[t] = argmax

        if cfg.cluster_threshold is not None and cfg.store_full_snapshots:
            F_snaps.append(F.copy())

        if diff < cfg.tol:
            argmax_history = argmax_history[: t + 1]
            if cfg.cluster_threshold is not None and cfg.store_full_snapshots:
                F_snaps = F_snaps[: t + 1]
            break

    F_final = F.copy()

    # final safe argmax and hard labels
    row_max_final = F_final.max(axis=1)
    argmax_final = F_final.argmax(axis=1)
    argmax_final[row_max_final <= 1e-12] = -1

    hard_labels = np.full(n, -1, dtype=np.int64)
    nz = row_max_final > 0
    if np.any(nz):
        hard_labels[nz] = classes[argmax_final[nz]]

    # layers: first t where argmax matches final; ignore t=0 for non-cores
    layer_idx = np.full(n, -1, dtype=int)
    t_start = max(0, int(cfg.start_t_for_noncore))
    for v in range(n):
        if core_mask[v]:
            layer_idx[v] = 0
            continue

        fl = argmax_final[v]
        if fl == -1:
            # never acquired label mass; assign to last
            layer_idx[v] = t_last
            continue

        if cfg.cluster_threshold is None:
            # basic temporal absorption
            ts = np.flatnonzero(argmax_history[t_start:, v] == fl)
            layer_idx[v] = (ts[0] + t_start) if ts.size > 0 else t_last
        else:
            # confidence-aware stabilization (requires value lookups)
            # If snapshots were not stored, we can't check the threshold; fall back
            if not cfg.store_full_snapshots:
                ts = np.flatnonzero(argmax_history[t_start:, v] == fl)
                layer_idx[v] = (ts[0] + t_start) if ts.size > 0 else t_last
            else:
                assigned = False
                for tt in range(t_start, argmax_history.shape[0]):
                    if argmax_history[tt, v] == fl and F_snaps[tt][v, fl] >= float(cfg.cluster_threshold):
                        layer_idx[v] = tt
                        assigned = True
                        break
                if not assigned:
                    layer_idx[v] = t_last

    # remap to contiguous layers while keeping core at 0
    if cfg.contiguous_layers:
        unique_layers = np.unique(layer_idx)
        # Ensure 0 stays 0 by ordering unique values ascending
        layer_map = {old: new for new, old in enumerate(unique_layers)}
        layer_idx = np.array([layer_map[x] for x in layer_idx], dtype=int)

    return hard_labels, F_final, layer_idx


def label_spread_absorbing_fast(
    A_csr: sp.csr_matrix,
    y: np.ndarray,
    cfg: DiffusionConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast label spreading with absorbing cores + diffusion-defined layers.
    Returns: hard_labels (n,), F_final (n x K), layer_idx (n,)
    """
    _apply_thread_env(cfg.num_threads)

    if not sp.isspmatrix_csr(A_csr):
        A_csr = A_csr.tocsr(copy=False)

    n = A_csr.shape[0]
    y = np.asarray(y, dtype=np.int64)
    if y.shape[0] != n:
        raise ValueError("y must have length equal to A_csr.shape[0]")

    core_mask = (y >= 0)
    if not np.any(core_mask):
        # Nothing to propagate
        return y.copy(), np.zeros((n, 0), dtype=cfg.dtype), np.zeros(n, dtype=int)

    # classes and mapping (stable order)
    classes = np.unique(y[core_mask])
    K = int(classes.shape[0])
    lab2idx = {int(c): i for i, c in enumerate(classes)}

    # One-hot initialization
    F0 = np.zeros((n, K), dtype=cfg.dtype)
    labeled_idx = np.where(core_mask)[0]
    F0[labeled_idx, [lab2idx[int(lbl)] for lbl in y[labeled_idx]]] = 1.0

    return diffusion_layers(
        A_csr, F0, core_mask, classes=classes, cfg=cfg
    )


# ------------------------------
# High-level orchestration
# ------------------------------

def cluster_core(corespect_obj, res_t: Optional[float] = None):
    """
    Cluster only the core layer (layers_[0]) of the graph at resolution res_t.
    """
    G = corespect_obj.G
    layers = corespect_obj.layers_
    core_nodes = layers[0]

    if res_t is None:
        # choose_stopping_res expected to be available from your utils
        res_t = choose_stopping_res(G, core_nodes)

    labels = cluster_subset(G, core_nodes, res_t)
    return labels


def majority(corespect_obj, cfg: Optional[MajorityConfig] = None):
    """
    High-level routine combining core clustering, majority pass, and diffusion layering.
    """
    from collections import Counter

    if cfg is None:
        cfg = MajorityConfig()

    print(cfg)

    if cfg.cluster_resolution is None:
        cfg = replace(cfg, cluster_resolution=2)

    # 1) Core clustering to get initial labels for core nodes
    final_labels = cluster_core(corespect_obj, cfg.cluster_resolution)

    layers = corespect_obj.layers_
    rem_nodes = [node for node in layers[1] if final_labels[node] == -1]
    G = corespect_obj.G
    print("Remaining nodes after core clustering:", len(rem_nodes))

    # 2) Adjacency (directional)
    Adj = igraph_to_sparse(G, direction=cfg.direction)

    # 3) Optional majority stage over remaining nodes (external dependency)
    # partitioned_majority_stage1 expected from your utils; keep as in original
    cnum = len(set(final_labels)) - 1
    _, final_labels, remaining_nodes = partitioned_majority_stage1(
        G, cnum, final_labels, np.array(rem_nodes).astype(int), 0
    )
    print("Label counts after majority stage:", Counter(final_labels))

    # 4) Diffusion + layers
    hard, F, layer_idx = label_spread_absorbing_fast(
        Adj, final_labels, cfg.diffusion
    )
    print("Layer stats:", int(np.min(layer_idx)), int(np.max(layer_idx)), len(set(layer_idx)))

    # 5) Build new_layers on the object
    n_layers = int(layer_idx.max()) + 1
    new_layers: List[List[int]] = [[] for _ in range(n_layers)]
    for i, li in enumerate(layer_idx):
        new_layers[li].append(i)
    corespect_obj.new_layers = new_layers

    return hard
