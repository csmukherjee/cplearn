from . import louvain_utils
from collections import deque

def louvain(G,resolution=1.0):



#    Gp=densify.Densify_v0(G, Gp, core_nodes, ng_num)
    total_partition= louvain_utils.louvain_partitions(G, weight="weight", resolution=resolution)
    partition_ = deque(total_partition, maxlen=1).pop()
    label_map= louvain_utils.partition_to_label(partition_)


    #Returns a dictionary of the labels for each node.
    return label_map
