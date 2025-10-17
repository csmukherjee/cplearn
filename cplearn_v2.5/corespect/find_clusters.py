#First finish this.

from ..utils.stable_core_extraction import *

def cluster_core(corespect_obj):
    G=corespect_obj.G
    layers=corespect_obj.layers_
    core_nodes=layers[0]

    if corespect_obj.fine_grained_res is None:
        res_t=choose_stopping_res(G,core_nodes)
    else:
        res_t=corespect_obj.fine_grained_res

    #res_t=choose_stopping_res(G,core_nodes)

    labels=cluster_subset(G,core_nodes,res_t)
    return labels

def majority(corespect_obj,find_cluster_params={}):

    #gmm_on_core=find_cluster_params.get('gmm_on_core',False)


    layers=corespect_obj.layers_

    final_labels=cluster_core(corespect_obj)
    rem_nodes=[]
    for node in layers[1]:
        if final_labels[node] == -1:
            rem_nodes.append(node)

    G=corespect_obj.G
    cnum=len(set(final_labels))-1

    print("Remaining nodes=",len(rem_nodes))

    curr_layers=layers[0]+layers[1]

    layer_num=1

    while layer_num<len(layers):
        _, final_labels, remaining_nodes = partitioned_majority_stage1(G, cnum, final_labels,
                                                                       np.array(rem_nodes).astype(int), 0)
        print("First try Remaining nodes=", remaining_nodes.shape, np.count_nonzero(final_labels == -1))

        if remaining_nodes.shape[0]>0:
            _, final_labels, remaining_nodes,_ = partitioned_majority_stage2(G, cnum, final_labels,
                                                                            remaining_nodes, 0)
            print("second try Remaining nodes=", remaining_nodes.shape, np.count_nonzero(final_labels == -1))

        layer_num += 1

        if layer_num<len(layers):
            rem_nodes = []
            curr_layers.extend(layers[layer_num])
            for node in curr_layers:
                if final_labels[node] == -1:
                    rem_nodes.append(node)




    return final_labels

def CDNN(X,layers,find_cluster_params=None):

    labels=None

    return labels
