
#Need graph building
#Need to call clustering on the core for partitioned majority

from joblib import Parallel, delayed
from numba import njit

import numpy as np
import leidenalg

import networkx as nx

from ..utils.gen_utils import get_kNN




from ..utils.clustering_algo import louvain


def cluster_subset(G,core_nodes,res):

    n=G.vcount()

    G_core = G.induced_subgraph(core_nodes)

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
    th=max(np.quantile(max_list, 0.75),0.1)

    print([np.quantile(max_list, 0.1*ell) for ell in range(10)],flush=True)


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
def partitioned_majority_stage2(G, cnum, proxy_labels, remaining_nodes):
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

        proxy_labels, remaining_nodes, added_nodes = partitioned_max_addition(edge_list,deg_list,n, cnum, proxy_labels, candidate_set_mask, eps=0)

        for ell in added_nodes:
            layer.append(ell)



        n1 = np.sum(proxy_labels != -1)


        if n1 - n0 < tolerance:
            remaining_nodes=candidate_set_mask.nonzero()[0]

            return layer,proxy_labels,remaining_nodes



        if n0 > 0.95 * G.vcount():
            remaining_nodes =candidate_set_mask.nonzero()[0]

            return layer,proxy_labels,remaining_nodes

        round_num += 1



#from collections import Counter

def find_one_layer_intersection(G,core_nodes,rem_nodes,eps,res):


    labels_init= cluster_subset(G, core_nodes,res=res)

    #print(Counter(labels_init),flush=True)


    if rem_nodes is not None:
        remaining_nodes_init = rem_nodes

    else:
        remaining_nodes_init = np.array(list(set([i for i in range(G.vcount())]) - set(core_nodes))).astype(int)

    cnum = len(set(labels_init)) - 1
    layer, proxy_labels_init, remaining_nodes_init = partitioned_majority_stage1(G, cnum, labels_init,
                                                                                 remaining_nodes_init, eps)

    return layer

def find_two_layers_intersection(G,core_nodes,rem_nodes,eps,res):

    labels_init= cluster_subset(G, core_nodes, res=res)

    if rem_nodes is not None:
        remaining_nodes_init = rem_nodes

    else:
        remaining_nodes_init = np.array(list(set([i for i in range(G.vcount())]) - set(core_nodes))).astype(int)

    cnum = len(set(labels_init)) - 1
    layer1, proxy_labels_init, remaining_nodes_init = partitioned_majority_stage1(G, cnum, labels_init,
                                                                                  remaining_nodes_init, eps)

    #Remaining nodes need to be updated to be reflected here.

    layer2, proxy_labels_init, remaining_nodes_init = partitioned_majority_stage2(G, cnum, proxy_labels_init,
                                                                                  remaining_nodes_init)
    layer = layer1 + layer2

    return layer



def find_intersection_layers(G, core_layer,rem_nodes=None,eps=0,mode='one_layer',res=1.0):


    if mode == 'one_layer':
        layer_list = Parallel(n_jobs=10)(delayed(find_one_layer_intersection)(G,core_layer,rem_nodes,eps,res) for _ in range(10))


    elif mode == 'two_layers':
        layer_list = Parallel(n_jobs=10)(delayed(find_two_layers_intersection)(G,core_layer,rem_nodes,eps,res) for _ in range(10))

    else:
        raise KeyError(f"{mode} is not a valid mode for finding intersection layers")


    print([len(layer) for layer in layer_list])

    layer_candidate=set(layer_list[0])
    for i in range(1,10):
        layer_candidate = layer_candidate.intersection(set(layer_list[i]))


    return layer_candidate



#This is the method designed to show stability of points w.r.t. reproducibility specifically.
#Now, we can also have two modes here. Partition and structural.
def stable_rank(self,layer_extraction_params=None):

    final_score=self.FlowRank_score
    n=len(final_score)

    #Initiate parameters
    core_fraction=layer_extraction_params.get('core_fraction',0.2)
    purify=layer_extraction_params.get('purify',0)
    res=layer_extraction_params.get('resolution',1)



    #Get the core nodes
    sorted_nodes= sorted(final_score, key=lambda k: final_score[k], reverse=True)
    core_nodes=np.array(sorted_nodes[:int(core_fraction*n)]).astype(int)


    print("Number of core nodes",len(core_nodes))

    #Get the graph
    G=self.G

    eps = 0  # This is set to 0 for now. Implies normal majority. Can be tweaked later

    #Get First stable layer.
    layer_candidate=find_intersection_layers(G,core_nodes,eps=eps,mode='one_layer',res=res)

    # Include points from core.
    core_nodes_candidate = find_intersection_layers(G, core_nodes, rem_nodes=np.array(core_nodes).astype(int), eps=eps,
                                                    mode='one_layer', res=res)

    #print("Number of core nodes selected", len(core_nodes_candidate))

    print(len(core_nodes_candidate),len(layer_candidate))

    layer_candidate = list(core_nodes_candidate) + list(layer_candidate)

    print("\n First stable layer size=",len(layer_candidate),'\n')


    #Here we can provide a new hierarchy from start upto layer_candidate.

    Layers = [list(layer_candidate)]
    remaining_nodes = set([i for i in range(n)])
    remaining_nodes = remaining_nodes - (set(layer_candidate))


    curr_nodes=[]
    for layer in Layers:
        for node in layer:
            curr_nodes.append(node)

    #Get second stable layer. #This needs fixing.
    layer_candidate = find_intersection_layers(G, np.array(curr_nodes).astype(int),eps=eps,mode='two_layers',res=res)
    Layers.append(list(layer_candidate))

    print("\n Second stable layer size=",len(layer_candidate),'\n')

    #The remaining is the last layer.
    remaining_nodes=remaining_nodes-(set(layer_candidate))
    layer=[]
    for i in remaining_nodes:
        layer.append(i)

    Layers.append(layer)



    return Layers