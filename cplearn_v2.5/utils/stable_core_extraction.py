from joblib import Parallel, delayed
import numpy as np
import leidenalg
from .densify import densify_v0


def choose_stopping_res(G,core_nodes,thr=0.95,densify=False,adj_list=False):

    n0=len(core_nodes)

    n1=n0
    t=1
    while n1>thr*n0 and t<5:
        layer_candidate=find_intersection_layers(G,core_layer=core_nodes,rem_nodes=core_nodes,res=t,densify=densify,adj_list=adj_list)
        n1=len(layer_candidate)
        print(t,n1,n0)
        t += 0.25

    return t-0.5


def cluster_subset(G,core_nodes,res,densify=False,adj_list=None):

    n=G.vcount()

    G_core = G.induced_subgraph(core_nodes)

    if densify:
        print("densifying...")
        G_core=densify_v0(adj_list,G,G_core,core_nodes)


    partition = leidenalg.find_partition(
        G_core,
        leidenalg.RBConfigurationVertexPartition,  # modularity-based partition
        resolution_parameter=res, # <-- control resolution here
        seed = np.random.randint(int(1e9))
    )

    # Map back to original node ids
    id_map = np.array(G_core.vs["id"], dtype=int)

    # Fill label vector using membership vector
    labels = -1 * np.ones(n, dtype=int)
    labels[id_map] = partition.membership
    return labels


def partitioned_majority_addition(edge_list,deg_list,n, cnum, final_labels, candidate_set_mask, eps=0):


    votes=np.zeros((n,cnum))
    added_nodes=[]

    for u,v in edge_list:

        c_idx=int(final_labels[v])
        if candidate_set_mask[u]==1 and c_idx!=-1:
                votes[u,c_idx]+=1


    for u in range(n):

        if candidate_set_mask[u]==1:
            if max(votes[u])>(0.5+eps)*deg_list[u]:
                final_labels[u]=np.argmax(votes[u])
                candidate_set_mask[u]=0
                added_nodes.append(u)



    return final_labels,candidate_set_mask,added_nodes


#This is trickier.
def partitioned_max_addition(edge_list,deg_list,n, cnum, final_labels, candidate_set_mask, eps=0):
    votes = np.zeros((n, cnum))
    added_nodes = []

    for u, v in edge_list:

        c_idx = int(final_labels[v])
        if candidate_set_mask[u] == 1 and c_idx != -1:
            votes[u, c_idx] += 1

    norm_votes=votes/deg_list.reshape(-1, 1)

    max_list=np.max(norm_votes,axis=1)
    nonzero_vals = max_list[max_list != 0]
    th = np.quantile(nonzero_vals, 0.75)
#    th=np.quantile(max_list, 0.75)

    #print([np.quantile(max_list, 0.1*ell) for ell in range(10)],flush=True)


    for u in range(n):

        if candidate_set_mask[u] == 1:
            if max(votes[u]) > th * deg_list[u]:
                final_labels[u] = np.argmax(votes[u])
                candidate_set_mask[u] = 0
                added_nodes.append(u)


    return final_labels,candidate_set_mask,added_nodes




#Control the first layer here.
def partitioned_majority_stage1(G, cnum, proxy_labels,remaining_nodes, eps):
    round_num=0
    layer=[]
    tolerance=int(0.005*G.vcount())
    n=G.vcount()

    edge_list=G.get_edgelist()
    deg_list=np.array(G.outdegree())
    candidate_set_mask=np.zeros(n).astype(int)
    candidate_set_mask[remaining_nodes]=1


    while True:
        n0 = np.sum(proxy_labels != -1)
        proxy_labels, candidate_set_mask,added_nodes = partitioned_majority_addition(edge_list,deg_list,n, cnum, proxy_labels, candidate_set_mask, eps=eps)

        for ell in added_nodes:
            layer.append(ell)

        n1 = np.sum(proxy_labels != -1)

        if n1 - n0 < tolerance:

            remaining_nodes=candidate_set_mask.nonzero()[0]

            return layer,proxy_labels,remaining_nodes


        if n0 > 0.95 * G.vcount():
            remaining_nodes = candidate_set_mask.nonzero()[0]
            return layer,proxy_labels,remaining_nodes

        round_num += 1





#Control the second layer here.
def partitioned_majority_stage2(G, cnum, proxy_labels, remaining_nodes,eps=0):
    round_num=0
    layer=[]
    tolerance=int(0.005*G.vcount())
    n=G.vcount()

    edge_list=G.get_edgelist()
    deg_list=np.array(G.outdegree())
    candidate_set_mask=np.zeros(n).astype(int)
    candidate_set_mask[remaining_nodes]=1

    layer_list=[]

    while True:
        n0 = np.sum(proxy_labels != -1)

        proxy_labels, candidate_set_mask, added_nodes = partitioned_max_addition(edge_list,deg_list,n, cnum, proxy_labels, candidate_set_mask, eps=eps)

        for ell in added_nodes:
            layer.append(ell)

        layer_list.append(added_nodes)

        n1 = np.sum(proxy_labels != -1)


        if n1 - n0 < tolerance:
            remaining_nodes=candidate_set_mask.nonzero()[0]

            return layer,proxy_labels,remaining_nodes,layer_list



        if n0 > 0.95 * G.vcount():
            remaining_nodes =candidate_set_mask.nonzero()[0]

            return layer,proxy_labels,remaining_nodes,layer_list

        round_num += 1




#from collections import Counter

def find_one_layer_intersection(G,core_nodes,rem_nodes,eps,res,densify,adj_list):


    labels_init= cluster_subset(G, core_nodes,res=res,densify=densify,adj_list=adj_list)

    #print(Counter(labels_init),flush=True)


    if rem_nodes is not None:
        remaining_nodes_init = rem_nodes

    else:
        remaining_nodes_init = np.array(list(set([i for i in range(G.vcount())]) - set(core_nodes))).astype(int)

    cnum = len(set(labels_init)) - (1 if -1 in labels_init else 0)
    layer, proxy_labels_init, remaining_nodes_init = partitioned_majority_stage1(G, cnum, labels_init,
                                                                                 remaining_nodes_init, eps)

    return layer

def find_two_layers_intersection(G,core_nodes,rem_nodes,eps,res,densify,adj_list):

    labels_init= cluster_subset(G, core_nodes,res=res,densify=densify,adj_list=adj_list)

    if rem_nodes is not None:
        remaining_nodes_init = rem_nodes

    else:
        remaining_nodes_init = np.array(list(set([i for i in range(G.vcount())]) - set(core_nodes))).astype(int)

    cnum = len(set(labels_init)) - 1
    layer1, proxy_labels_init, remaining_nodes_init = partitioned_majority_stage1(G, cnum, labels_init,
                                                                                  remaining_nodes_init, eps)

    #Remaining nodes need to be updated to be reflected here.

    layer2, proxy_labels_init, remaining_nodes_init,layer_list = partitioned_majority_stage2(G, cnum, proxy_labels_init,
                                                                                  remaining_nodes_init)
 #   layer = layer1 + layer2
 #   return layer
    f_layer_list=[layer1]
    f_layer_list.extend(layer_list)

    return f_layer_list


def merge_until_threshold(list_set, n,size_th_frac=0.1):
    threshold = int(size_th_frac * n)
    merged, cur = [], []
    cur_len = 0

    for lst in list_set:
        cur.extend(lst)
        cur_len += len(lst)
        if cur_len >= threshold:
            merged.append(cur)
            cur, cur_len = [], 0

    if cur:
        merged.append(cur)

    # Ensure last block also meets threshold (if possible)
    if len(merged) > 1 and len(merged[-1]) < threshold:
        merged[-2].extend(merged[-1])
        merged.pop()

    return merged


#Here we use the list of lists to obtain a cleaner list. This is a fast process.
def process_list_of_layers(list_of_layer_list,n):

    Layer_list=[]
    iter_num=len(list_of_layer_list)

    cur_layer=[[] for _ in range(iter_num)]
    t=0
    c=0
    while True:

        for ell in range(0,iter_num):
            if t<len(list_of_layer_list[ell]):
                cur_layer[ell].extend(list_of_layer_list[ell][t])

            else:
                c+=1

        sets = [set(lst) for lst in cur_layer]

        # Intersect all
        common = set.intersection(*sets)
        Layer_list.append(list(common))

        for ell in range(iter_num):
            cur_layer[ell]=[x for x in cur_layer[ell] if x not in common]

        t+=1
        if c>=iter_num:

            Layer_list=merge_until_threshold(Layer_list,n)

            return Layer_list





def find_intersection_layers(G, core_layer,rem_nodes=None,eps=0,mode='one_layer',res=1.0,densify=False,adj_list=None):


    if mode == 'one_layer':
        layer_list = Parallel(n_jobs=10)(delayed(find_one_layer_intersection)(G,core_layer,rem_nodes,eps,res,densify,adj_list) for _ in range(10))
        #print([len(layer) for layer in layer_list])

        layer_candidate=set(layer_list[0])
        for i in range(1,10):
            layer_candidate = layer_candidate.intersection(set(layer_list[i]))

        return layer_candidate

    elif mode == 'two_layers':
        list_of_layer_list = Parallel(n_jobs=10)(delayed(find_two_layers_intersection)(G,core_layer,rem_nodes,eps,res,densify,adj_list) for _ in range(10))


        Layer_list=process_list_of_layers(list_of_layer_list,G.vcount())


        return Layer_list


    else:
        raise KeyError(mode," is not a valid mode for finding intersection layers")


