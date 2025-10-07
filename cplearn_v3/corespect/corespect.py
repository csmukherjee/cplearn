from .ranking import FlowRank
from . import layer_extraction
from . import find_clusters

from ..utils.gen_utils import get_kNN
from ..utils.gen_utils import get_edge_list

import igraph as ig


#To-do: Automatic selection of core_fraction, q, and r (Rank outputs)
def _extract_layers(self, method='quantile', layer_extraction_params=None):

    #Parameters for ranking algorithm
    #Initial ranking of points using our decided ranking algorithm.
    if layer_extraction_params is None:
        layer_extraction_params = dict()

    q = layer_extraction_params.get('q', 20)
    r = layer_extraction_params.get('r', 20)

    knn_list,knn_dist=get_kNN(self.X, q)

    #Initiage iGraph
    e_list = get_edge_list(knn_list,self.X.shape[0])
    G = ig.Graph(directed=True)
    G.add_vertices(self.X.shape[0])
    G.add_edges(e_list)
    G.vs["id"] = list(range(G.vcount()))


    self.knn_list=knn_list
    self.knn_dists=knn_dist
    self.G=G



    final_score = FlowRank(knn_list, r)

    self.FlowRank_score=final_score

    print("cp1")

    if hasattr(layer_extraction,method):
        func = getattr(layer_extraction, method)
        if callable(func):
            layers = func(self,layer_extraction_params)

        else:
            raise TypeError(f"{method} is not callable")
    else:
        raise KeyError(f"{method} is not a valid layer extraction method")


    return layers

#We are currently using Louvain for clustering. Should add Louvain.
#Propagation methods: {'majority','CDNN'}
def _find_clusters(self,prop_method,find_cluster_params=None):

    if hasattr(find_clusters,prop_method):
        func = getattr(find_clusters,prop_method)
        if callable(func):
            labels_ = func(self.X,self.layers,find_cluster_params)

        else:
            raise TypeError(f"{prop_method} is not callable")
    else:
        raise KeyError(f"{prop_method} is not a valid propagation method")

    return labels_


class Corespect:
    def __init__(self, X=None):

        self.X = X
        self.n = X.shape[0]

        self.knn_list = None
        self.knn_dists = None
        self.G = None
        self.FlowRank_score=None


        self.layers_=None
        self.labels_=None

        self.set_labels=None
        self.set_layers=None



    # Layer extraction methods: {'quantile','partitioned_majority','majority','stable_rank'}
    def extract_layers(self, method='partitioned_majority', core_fraction=0.15, layer_extraction_params=None):

        layers_= _extract_layers(self, method=method,
                                 layer_extraction_params=layer_extraction_params)
        self.layers_=layers_

    # Propagation methods: {'majority','CDNN'}
    def find_clusters(self,prop_method='majority',find_cluster_params=None):


        labels_=_find_clusters(self,prop_method=prop_method,find_cluster_params=find_cluster_params)
        self.labels_=labels_

    def container(self):
        self.set_labels.append()
        self.set_layers.append()


    def get_layers_and_labels(self):
        return self.layers_,self.labels_

