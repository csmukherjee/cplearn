#Densification has not been used yet.
from kneed import KneeLocator
import igraph as ig

from ..utils.gen_utils import get_kNN,get_edge_list,get_igraph_from_knn_list,get_induced_knn
from ..utils.stable_core_extraction import *

from .ranking import FlowRank

from matplotlib import pyplot as plt

import numpy as np

def cutoffs(values, eps=1/400, mass=0.99, win=50, rel_slope=0.01):
    v = np.asarray(values, float)
    # assume v is already sorted desc
    out = {}

    # 1) last non-zero
    nz = np.argmax(v[::-1] > 0)
    out["last_nonzero"] = len(v) - nz if v[-1] == 0 else len(v)

    # 2) absolute threshold
    i_eps = np.searchsorted(-v, -eps, side="left")
    out["value>eps"] = i_eps

    # 3) cumulative mass
    cs = np.cumsum(v)
    total = cs[-1] if cs[-1] > 0 else 1.0
    i_mass = np.searchsorted(cs, mass * total, side="left") + 1
    out["cumu_mass"] = i_mass

    # 4) knee (simple 2-line distance method)
    x = np.arange(len(values))
    y = values

    knee = KneeLocator(x, y, curve='convex', direction='decreasing')
    knee_point = knee.knee
    out["knee"] = knee_point

    # 5) flat-tail via relative slope
    vp=v[int(0.05*len(v)):]
    dv = np.abs(np.diff(vp, prepend=vp[0]))
    # rolling mean of |Δv|/v
    rel = (dv / np.maximum(vp, 1e-12))
    rel_roll = np.convolve(rel, np.ones(win)/win, mode="same")
    flat_idx = np.argmax(rel_roll < rel_slope)  # first True
    out["flat_tail"] = int(flat_idx) if rel_roll[flat_idx] < rel_slope else len(v)

    return out




#Add partitioned majority
def partitioned_majority():

    return None


#This is the method designed to show stability of points w.r.t. reproducibility specifically.
#Now, we can also have two modes here. Partition and structural.
def stable_rank(self,layer_extraction_params=None):

    final_score=self.FlowRank_score
    n=len(final_score)

    #Initiate parameters
    core_fraction=layer_extraction_params.get('core_fraction',0.1)

    early_cutoff=layer_extraction_params.get('early_cutoff',False)

    if early_cutoff:
        core_selection_mode="knee"
    else:
        core_selection_mode="cumu_mass"



    res=layer_extraction_params.get('resolution',1)
    self.resolution=res


    densify=layer_extraction_params.get('densify',False)
    fine_grained =layer_extraction_params.get('fine_grained',False)
    knn_list_n=np.array(self.knn_list).astype(int)

    q=layer_extraction_params.get('q',20)
    r=layer_extraction_params.get('r',10)


    plt.figure(figsize=(8, 6))
    values = sorted(final_score.values(), reverse=True)
    cuts=cutoffs(values)
    for name, idx in cuts.items():
        print(f"{name:>12} : {idx}")
    plt.plot(values, marker='o', linewidth=2)
    plt.title('FlowRank values')
    plt.show()


    #Get the core nodes
    sorted_nodes= sorted(final_score, key=lambda k: final_score[k], reverse=True)
    #core_nodes=np.array(sorted_nodes[:int(core_fraction*n)]).astype(int)

    #Change: Using cutoff values:
    core_nodes=np.array(sorted_nodes[:int(min(cuts[core_selection_mode],0.25*n))]).astype(int)


    print("Number of core nodes",len(core_nodes))

    #Get the graph
    G=self.G

    eps = 0  # This is set to 0 for now. Implies normal majority. Can be tweaked later

    #Get First stable layer.
    layer_candidate=find_intersection_layers(G,core_nodes,eps=eps,mode='one_layer',res=res,densify=densify,adj_list=knn_list_n)

    # Include points from core.
    core_nodes_candidate = find_intersection_layers(G, core_nodes, rem_nodes=np.array(core_nodes).astype(int), eps=eps,
                                                    mode='one_layer', res=res,densify=densify,adj_list=knn_list_n)


    layer_candidate = list(core_nodes_candidate) + list(layer_candidate)


    #Here we can provide a new hierarchy from start upto layer_candidate.

    Layers = [list(layer_candidate)]
    remaining_nodes = set([i for i in range(n)]) - (set(layer_candidate))


    # curr_nodes=list(layer_candidate)
    #
    # New_set_of_layers = find_intersection_layers(G, np.array(curr_nodes).astype(int),eps=eps,mode='two_layers',res=res,densify=densify,adj_list=knn_list_n)
    # for layer in New_set_of_layers:
    #     Layers.append(layer)
    #     remaining_nodes = remaining_nodes - (set(layer))


    layer=[]
    for i in remaining_nodes:
        layer.append(i)

    Layers.append(layer)


    #ToDo: Write From Scratch. Be careful about igraph implementation and effect.
    if fine_grained:
        stable_core=Layers[0]
        n1=len(stable_core)

        #next_core_fraction=0.15 #This is random.
        # X_stable = self.X[stable_core]
        # n1 = X_stable.shape[0]
        # knn_list_stable, knn_dists_stable = get_kNN(X_stable,q=q)

        knn_list_stable, knn_dists_stable=get_induced_knn(knn_list_n,self.knn_dists,stable_core,n)

        knn_list_stable_plain = [list(row) for row in knn_list_stable]
        knn_dists_stable_plain = [list(row) for row in knn_dists_stable]


        stable_score = FlowRank(knn_list_stable, r)
        sorted_nodes = sorted(stable_score, key=lambda k: stable_score[k], reverse=True)


        plt.figure(figsize=(8, 6))
        values = sorted(stable_score.values(), reverse=True)

        cuts = cutoffs(values)
        for name, idx in cuts.items():
            print(f"{name:>12} : {idx}")

        plt.plot(values, marker='o', linewidth=2)
        plt.title('FlowRank values on the stable core')
        plt.show()


        #ToDo: Here the code is very unclean.
        # It is better to convert the whole problem to a subset, and then revert back.

        G_stable=get_igraph_from_knn_list(knn_list_stable,len(stable_core))





        # e_list = get_edge_list(knn_list_stable, n1)
        # G_stable = ig.Graph(directed=True)
        # G_stable.add_vertices(n1)
        # G_stable.add_edges(e_list)
        # G_stable.vs["id"] = list(range(n1))

        new_core = np.array(sorted_nodes[:int(core_fraction * n)]).astype(int)

        # Change: Using cutoff values:
        #new_core=np.array(sorted_nodes[:int(cuts[core_selection_mode])]).astype(int)


        res_t = choose_stopping_res(G_stable, new_core)
        self.fine_grained_res=res_t

        stable_new_core = find_intersection_layers(G_stable, new_core,
                                                       rem_nodes=np.array(new_core).astype(int), eps=0,
                                                       mode='one_layer', res=res_t, densify=densify,
                                                       adj_list=knn_list_stable_plain)



        print(len(stable_new_core),len(new_core))

        rem_nodes=list(set([i for i in range(n1)])-set(stable_new_core))
        labels_init = cluster_subset(G_stable, stable_new_core, res=res_t, densify=densify, adj_list=knn_list_stable_plain)
        cnum=len(set(labels_init))-1

        next_layer,proxy_labels,remaining_nodes=partitioned_majority_stage1(G_stable, cnum, labels_init,
                                    np.array(rem_nodes).astype(int), eps)


        rem_nodes=list(set(remaining_nodes)-set(next_layer))

        new_Layers_init=[stable_new_core,next_layer,rem_nodes]

        new_Layers=[]
        for layer in new_Layers_init:
            new_layer=[]
            for node in layer:
                new_layer.append(stable_core[node])

            new_Layers.append(new_layer)



        for i in range(1,len(Layers)):
            new_Layers.append(Layers[i])

        Layers=new_Layers

        #ToDo: Fix this later. This is not the right way.
        # Update k-nn list. This is first fix for coremap. #Should we have kept the original neighbors as well?

        # for i, node in enumerate(stable_core):
        #     self.knn_list[node] = [stable_core[ell] for ell in knn_list_stable[i]]
        #     self.knn_dists[node] =knn_dists_stable[i]
        #
        # #Updating the graph directly. Slightly costly.
        # G=get_igraph_from_knn_list(self.knn_list,n)
        # G.vs["id"] = list(range(G.vcount()))
        # self.G=G

    print("Completed with layer size:")
    for layer in Layers:
        print(len(layer),end=' ')
    print('\n')




    return Layers




