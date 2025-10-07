import time
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def top_nonzero_indices_fixed(mat, f, K):
    m, n = mat.shape
    out = -np.ones((m, K), dtype=np.int64)

    for i in prange(m):
        # collect nonzeros
        nz_idx = []
        nz_vals = []
        for j in range(n):
            v = mat[i, j]
            if v != 0:
                nz_idx.append(j)
                nz_vals.append(v)

        nnz = len(nz_idx)
        if nnz == 0:
            continue

        k = f[i] if f[i] < nnz else nnz
        # sort top-k
        order = np.argsort(np.array(nz_vals))[::-1][:k]
        sel = np.array(nz_idx)[order]

        out[i, :k] = sel  # write results, rest stay -1
    return out


import numpy as np
from numba import njit, prange

@njit(parallel=True)
def topk_indices_per_row(mat, k):
    """
    mat: 2D array (m, n)
    k: number of top indices to extract per row
    returns: (m, k) array of indices sorted by descending values
             (ties broken by index order)
    """
    m, n = mat.shape
    out = np.empty((m, k), dtype=np.int64)

    for i in prange(m):
        row = mat[i]
        # argsort descending, then take first k
        order = np.argsort(row)[::-1][:k]
        out[i] = order
    return out



@njit(parallel=True, cache=True)
def batch_random_walk(adj_matrix, core_nodes, steps=5):
    """
    adj_matrix: (n, k) int array, each row has exactly k neighbors
    core_nodes: list of starting nodes
    steps: number of walk steps
    returns: (len(core_nodes), n) cumulative visitation distributions
    """
    n, k = adj_matrix.shape
    m = len(core_nodes)
    mat = np.zeros((m, n), dtype=np.float32)
    k_inv = 1.0 / k

    for i in prange(m):                 # parallel over seeds
        u = core_nodes[i]

        pi = np.zeros(n, dtype=np.float32)
        pi[u] = 1.0
        visits = np.zeros(n, dtype=np.float32)  # cumulative footprint
        visits[u] = 1.0

        new_pi = np.zeros(n, dtype=np.float32)

        for _ in range(steps):
            new_pi.fill(0.0)
            for v in range(n):
                pv = pi[v]
                if pv != 0.0:
                    contrib = pv * k_inv
                    row = adj_matrix[v]
                    for j in range(k):
                        vp = row[j]
                        new_pi[vp] += contrib
            pi, new_pi = new_pi, pi
            visits += pi   # accumulate visits at this step

        mat[i] = visits
    return mat



def densify_v0(adj_list,G,H,top_nodes):

    n=G.vcount()
    n_core=H.vcount()

    t0=time.time()

    original_deg= np.array(G.degree(mode="OUT"))[top_nodes]
    new_deg=np.array(H.degree(mode="OUT"))

    hmap=np.zeros(n)
    for i,u in enumerate(top_nodes):
        hmap[u]=i


    set_to_expand=[]
    vacancy=[]

    for i in range(n_core):
        if new_deg[i]<0.75*original_deg[i]:
            set_to_expand.append(i)
            vacancy.append(int(original_deg[i]-new_deg[i]))

    set_to_expand=np.array(set_to_expand).astype(int)

#    print(f"{len(vacancy)} vertices appended out of {len(top_nodes)}")

    t1=time.time()

    mat=batch_random_walk(adj_list, set_to_expand)

    t2=time.time()
    # This can be significantly improved.
    #results=top_nonzero_indices_fixed(mat, vacancy,max(vacancy))

    results=topk_indices_per_row(mat, max(vacancy))

    print(results.shape)

    t3=time.time()


    e_list=[]
    c=0
    for u in range(len(results)):

        c_temp=0

        for i,v in enumerate(results[u]):

            if mat[u,i]==0:
                break

            if c_temp== vacancy[u]:
                break

            if hmap[v] != 0:
                e_list.append((int(hmap[u]),int(hmap[v])))
                c_temp+=1
                c+=1

    H.add_edges(e_list)

    t4=time.time()

    print(f"Time for step 1: {t1-t0:.3f}")
    print(f"Time for step 2: {t2-t1:.3f}")
    print(f"Time for step 3: {t3-t2:.3f}")
    print(f"Time for step 4: {t4-t3:.3f}")
    #
    # print(f"A total of {c} edges were added")

    return H