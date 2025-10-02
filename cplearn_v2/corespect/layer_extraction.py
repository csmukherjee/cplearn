
#Need graph building
#Need to call clustering on the core for partitioned majority

from joblib import Parallel, delayed

import numpy as np
import networkx as nx

from ..utils.gen_utils import get_kNN
from ..utils.clustering_algo import louvain

#Just partition the ranked list into quantiles, and add nodes based on that.
def quantile(X,final_score):

    n=X.shape[0]
    sorted_nodes= sorted(final_score, key=lambda k: final_score[k], reverse=True)

    assert len(sorted_nodes)==n, "Length of sorted nodes should be equal to n"
    Layers=[]
    for q in range(0,10):
        Layers.append(sorted_nodes[int(q*n/10):int((q+1)*n/10)])


    return Layers


def get_louvain_partition(G, core_nodes):
    hmap = np.zeros(G.number_of_nodes())

    G_core = nx.DiGraph()
    for ell in core_nodes:
        hmap[ell] = 1
        G_core.add_node(ell)

    for u, v in G.edges():
        if hmap[u] == 1 and hmap[v] == 1:
            G_core.add_edge(u, v, weight=G[u][v]['weight'])

    label_map = louvain(G_core)
    return label_map

def initiate_stable_rank_env(G,core_nodes):

    n=G.number_of_nodes()
    remaining_nodes = set([i for i in range(n)]) - set(core_nodes)
    label_map = get_louvain_partition(G, core_nodes)

    proxy_labels=-1*np.ones(n).astype(int)
    for ell in core_nodes:
        proxy_labels[ell] = label_map[ell]

    return proxy_labels,remaining_nodes

def initiate_majority_env(X,core_nodes,layer_extraction_params=None,partition=1):

    n=X.shape[0]
    ng_num = layer_extraction_params.get('ng_num', 20)

    Layers=[]


    remaining_nodes=set([i for i in range(n)])-set(core_nodes)

    # Set as Layer 0
    layer0=[]
    for ell in core_nodes:
        layer0.append(ell)
    Layers.append(layer0)


    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)

    knn_list, _ = get_kNN(X, q=ng_num)
    for i in range(n):
        for j in knn_list[i, :]:
            if i != j:
                G.add_edge(i, j, weight=1)

    proxy_labels=-1*np.ones(n).astype(int)


    if partition==1:
        print("proxy label has clusters")
        label_map=get_louvain_partition(G, core_nodes)
        for ell in core_nodes:
            proxy_labels[ell]=label_map[ell]


    else:
        for ell in core_nodes:
            proxy_labels[ell]=0



    return G,proxy_labels,remaining_nodes,Layers


def partitioned_majority_addition_weighted(G, cnum, final_labels, current_nodes, remaining_nodes, eps=0):

    #print("Remaining nodes in this round",len(remaining_nodes))

    n=G.number_of_nodes()

    c_deg = np.zeros(cnum)
    c_edges=np.zeros(cnum)

    hmap=-1*np.ones(n)

    for ell in current_nodes:
        hmap[ell]=0

    for u,v in G.edges():
        if hmap[u]==0 and hmap[v]==0:
            c_edges[final_labels[u]]+=1


    added_nodes=set()
    for ell in remaining_nodes:
        votes=np.zeros(cnum)
        for u in G.neighbors(ell):
            if final_labels[u]!=-1:
                    votes[final_labels[u]]+=1

        if max(votes)> (0.5+eps)*G.out_degree(ell):
            final_labels[ell]=np.argmax(votes)
            added_nodes.add(ell)

    #print("check here",len(added_nodes))
    remaining_nodes=remaining_nodes-added_nodes

    return final_labels,remaining_nodes,added_nodes


def partitioned_majority_addition(G, cnum, final_labels, remaining_nodes, eps=0):

    #print("Remaining nodes in this round",len(remaining_nodes))

    added_nodes=set()
    for ell in remaining_nodes:
        votes=np.zeros(cnum)
        for u in G.neighbors(ell):
            if final_labels[u]!=-1:
                    votes[final_labels[u]]+=1

        if max(votes)> (0.5+eps)*G.out_degree(ell):
            final_labels[ell]=np.argmax(votes)
            added_nodes.add(ell)

    #print("check here",len(added_nodes))
    remaining_nodes=remaining_nodes-added_nodes

    return final_labels,remaining_nodes,added_nodes


def partitioned_max_addition(G, cnum, final_labels, remaining_nodes):

    added_nodes=set()



    #Calculate max.
    max_list=[]
    for ell in remaining_nodes:
        votes=np.zeros(cnum)
        for u in G.neighbors(ell):
            if final_labels[u]!=-1:
                    votes[final_labels[u]]+=1


        max_list.append(max(votes))


    #Add nodes with max votes greater than median of max_list
    for ell in remaining_nodes:
        votes = np.zeros(cnum)
        for u in G.neighbors(ell):
            if final_labels[u] != -1:
                votes[final_labels[u]] += 1


        thr= np.percentile(max_list, 75)

        if max(votes)> thr:
            final_labels[ell] = np.argmax(votes)
            added_nodes.add(ell)




    remaining_nodes=remaining_nodes-added_nodes

    return final_labels,remaining_nodes,added_nodes




#These can be called multiple times if needed (for different thresholds)

#Control the first layer here.
def partitioned_majority_stage1(G, cnum, proxy_labels,remaining_nodes, eps):
    round_num=0
    layer=[]
    tolerance=int(0.005*G.number_of_nodes())

    while True:
        n0 = np.sum(proxy_labels != -1)
        proxy_labels, remaining_nodes, added_nodes = partitioned_majority_addition(G, cnum, proxy_labels,
                                                                                       remaining_nodes, eps)

        #print("Round number=", round_num, " Nodes added=", len(added_nodes), " Remaining nodes=",len(remaining_nodes))

        for ell in added_nodes:
            layer.append(ell)


        n1 = np.sum(proxy_labels != -1)

        if n1 - n0 < tolerance:
            return layer,proxy_labels,remaining_nodes


        if n0 > 0.95 * G.number_of_nodes():
            return layer,proxy_labels,remaining_nodes

        round_num += 1




#Control the second layer here.
def partitioned_majority_stage2(G, cnum, proxy_labels, remaining_nodes):
    tolerance = int(0.005 * G.number_of_nodes())
    round_num=0
    layer=[]
    while True:
        n0 = np.sum(proxy_labels != -1)

        proxy_labels, remaining_nodes, added_nodes = partitioned_max_addition(G, cnum, proxy_labels, remaining_nodes)

        for ell in added_nodes:
            layer.append(ell)



        n1 = np.sum(proxy_labels != -1)


        if n1 - n0 < tolerance:
            return layer,proxy_labels,remaining_nodes



        if n0 > 0.95 * G.number_of_nodes():
            return layer,proxy_labels,remaining_nodes

        round_num += 1

#Partition=1 means we cluster core nodes, else all core nodes have same label.


#The first layer are the core nodes. We cluster them, and add points w.r.t. recursive majority first,
#And then w.r.t. adaptive majority.
#The last layer are the remaining nodes.
#We store the round_info internally, but currently do no return it. It can be returned if needed.
#Essentially there is a duplication of clustering effort down the line, but it is not that much.

#We can start with 0.9 as threshold as well, and go down to 0.5, giving a smoother transition and more layers.
#We should probably do that.

def partitioned_majority(X,final_score,core_fraction,layer_extraction_params=None):

    n=X.shape[0]

    #Get the core nodes
    sorted_nodes= sorted(final_score, key=lambda k: final_score[k], reverse=True)
    core_nodes=np.array(sorted_nodes[:int(core_fraction*n)]).astype(int)

    G, proxy_labels, remaining_nodes, Layers= initiate_majority_env(X,core_nodes,layer_extraction_params=layer_extraction_params,partition=1)

    cnum=len(set(proxy_labels))-1

    #This can control which thresholds we like.
    eps_scheduler=[0]
    #Get majority based layers:
    for eps in eps_scheduler:
        layer,proxy_labels,remaining_nodes=partitioned_majority_stage1(G,cnum,proxy_labels,remaining_nodes,eps)
        Layers.append(layer)

    #Get Layer 2
    layer,proxy_labels,remaining_nodes=partitioned_majority_stage2(G,cnum,proxy_labels,remaining_nodes)
    Layers.append(layer)

    layer=[]
    for i in range(n):
        if proxy_labels[i]==-1:
            layer.append(i)

    Layers.append(layer)

    return Layers


#--- Finish these two tomorrow--

#This is a more vanilla method in some sense, where we do not partition the core nodes, and just add nodes based on their
#Connections to the core nodes recursively. Here we will do it based on lowering thresholds.
#What if thresholds go up down the round? Sorting every round will be expensive. Figure this out tomorrow.
def structural_majority(X,final_score,core_fraction,layer_extraction_params=None):

    #Get the core nodes
    n=X.shape[0]
    sorted_nodes= sorted(final_score, key=lambda k: final_score[k], reverse=True)
    core_nodes=np.array(sorted_nodes[:int(core_fraction*n)]).astype(int)

    G, proxy_labels, remaining_nodes, Layers= initiate_majority_env(X,core_nodes,layer_extraction_params=layer_extraction_params,partition=0)

    cnum = len(set(proxy_labels)) - 1


    # This can control which thresholds we like.
    eps_scheduler = [0.3, 0.2, 0.1, 0]
    # Get majority based layers:
    for eps in eps_scheduler:
        layer, proxy_labels, remaining_nodes = partitioned_majority_stage1(G, cnum, proxy_labels, remaining_nodes, eps)
        Layers.append(layer)

    # Get Layer 2
    layer, proxy_labels, remaining_nodes = partitioned_majority_stage2(G, cnum, proxy_labels, remaining_nodes)
    Layers.append(layer)

    layer = []
    for i in range(n):
        if proxy_labels[i] == -1:
            layer.append(i)

    Layers.append(layer)
    return Layers


def cross_purify_core_one_round(G,core_nodes,rem_nodes=None,eps=0):
    proxy_labels_init,_=initiate_stable_rank_env(G, core_nodes)
    cnum = len(set(proxy_labels_init)) - 1

    _,_,refined_core=partitioned_majority_addition(G, cnum, proxy_labels_init, rem_nodes, eps=eps)
    return list(refined_core)

def cross_purify_core(G,core_nodes,rem_nodes,eps):

    if rem_nodes is None:
        rem_nodes = set([ell for ell in G.nodes()])

    layer_list = Parallel(n_jobs=10)(delayed(cross_purify_core_one_round)(G, core_nodes, rem_nodes, eps) for _ in range(10))
    core_nodes_candidate=set(layer_list[0])
    for i in range(1,10):
        core_nodes_candidate = core_nodes_candidate.intersection(set(layer_list[i]))




def find_one_layer_intersection(G,core_layer,rem_nodes,eps):
    proxy_labels_init, remaining_nodes_init = initiate_stable_rank_env(G, core_layer)

    if rem_nodes is not None:
        remaining_nodes_init = rem_nodes

    cnum = len(set(proxy_labels_init)) - 1
    layer, proxy_labels_init, remaining_nodes_init = partitioned_majority_stage1(G, cnum, proxy_labels_init,
                                                                                 remaining_nodes_init, eps)

    return layer

def find_two_layers_intersection(G,core_layer,rem_nodes,eps):
    proxy_labels_init, remaining_nodes_init = initiate_stable_rank_env(G, core_layer)

    if rem_nodes is not None:
        remaining_nodes_init = rem_nodes

    cnum = len(set(proxy_labels_init)) - 1
    layer1, proxy_labels_init, remaining_nodes_init = partitioned_majority_stage1(G, cnum, proxy_labels_init,
                                                                                  remaining_nodes_init, eps)
    layer2, proxy_labels_init, remaining_nodes_init = partitioned_majority_stage2(G, cnum, proxy_labels_init,
                                                                                  remaining_nodes_init)
    layer = layer1 + layer2

    return layer




    return core_nodes_candidate

def find_intersection_layers(G, core_layer,rem_nodes=None,eps=0,mode='one_layer'):


    if mode == 'one_layer':
        layer_list = Parallel(n_jobs=10)(delayed(find_one_layer_intersection)(G,core_layer,rem_nodes,eps) for _ in range(10))


    elif mode == 'two_layers':
        layer_list = Parallel(n_jobs=10)(delayed(find_two_layers_intersection)(G,core_layer,rem_nodes,eps) for _ in range(10))

    else:
        raise KeyError(f"{mode} is not a valid mode for finding intersection layers")


    layer_candidate=set(layer_list[0])
    for i in range(1,10):
        layer_candidate = layer_candidate.intersection(set(layer_list[i]))


    return layer_candidate



#This is the method designed to show stability of points w.r.t. reproducibility specifically.
#Now, we can also have two modes here. Partition and structural.
def stable_rank(X,final_score,core_fraction,layer_extraction_params=None):

    #Get the core nodes
    n=X.shape[0]
    sorted_nodes= sorted(final_score, key=lambda k: final_score[k], reverse=True)
    core_nodes=np.array(sorted_nodes[:int(core_fraction*n)]).astype(int)

    purify=layer_extraction_params.get('purify',0)

    print("Number of core nodes:",len(core_nodes))

    #Initialize environment
    G, proxy_labels, remaining_nodes_throwaway, init_Layers= initiate_majority_env(X,core_nodes,layer_extraction_params=layer_extraction_params,partition=1)

    eps = 0  # This is set to 0 for now. Implies normal majority. Can be tweaked later

    #Get First stable layer.
    layer_candidate=find_intersection_layers(G,core_nodes,eps=eps,mode='one_layer')
    print("\n First stable layer size=",len(layer_candidate),'\n')


    if purify ==1:
        #Include points from core.
        core_nodes_candidate=find_intersection_layers(G, core_nodes,rem_nodes=set(core_nodes),eps=eps,mode='one_layer')
        layer_candidate=list(core_nodes_candidate)+list(layer_candidate)
        print("New layer 0=",len(layer_candidate))

        # layer_candidate=find_intersection_layers(G, layer_candidate,rem_nodes=set(layer_candidate),eps=eps,mode='one_layer')
        # print("After second round of filtering:",len(layer_candidate),'\n')

        Layers = [list(layer_candidate)]
        remaining_nodes = set([i for i in range(n)])
        remaining_nodes = remaining_nodes - (set(layer_candidate))


    elif purify ==2:
        layer_candidate = find_intersection_layers(G, core_nodes, rem_nodes=set([i for i in range(n)]), eps=eps,
                                                     mode='one_layer')
        print("\n New stable layer size=", len(layer_candidate), '\n')

        Layers = [list(layer_candidate)]
        remaining_nodes = set([i for i in range(n)])
        remaining_nodes = remaining_nodes - (set(layer_candidate))



    else:
        Layers = [list(core_nodes),list(layer_candidate)]
        remaining_nodes=set([i for i in range(n)])
        remaining_nodes=remaining_nodes-(set(core_nodes))
        remaining_nodes=remaining_nodes-(set(layer_candidate))


    curr_nodes=[]
    for layer in Layers:
        for node in layer:
            curr_nodes.append(node)

    #Get second stable layer.
    layer_candidate = find_intersection_layers(G, curr_nodes,eps=eps,mode='two_layers')
    Layers.append(list(layer_candidate))

    print("\n Second stable layer size=",len(layer_candidate),'\n')

    #The remaining is the last layer.
    remaining_nodes=remaining_nodes-(set(layer_candidate))
    layer=[]
    for i in remaining_nodes:
        layer.append(i)

    Layers.append(layer)



    return Layers