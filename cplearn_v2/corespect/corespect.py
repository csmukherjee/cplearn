from .ranking import FlowRank
from . import layer_extraction
from . import find_clusters

#To-do: Automatic selection of core_fraction, q, and r (Rank outputs)
def _extract_layers(self, method='quantile', core_fraction=0.15, layer_extraction_params=None):

    #Parameters for ranking algorithm
    #Initial ranking of points using our decided ranking algorithm.
    if layer_extraction_params is None:
        layer_extraction_params = dict()
    ranking_algo_params={}
    if layer_extraction_params is not None:
        if 'q' in layer_extraction_params:
            ranking_algo_params['q']=layer_extraction_params['q']
        if 'r' in layer_extraction_params:
            ranking_algo_params['r']=layer_extraction_params['r']

    final_score=FlowRank(self.X,ranking_algo_params)

    if hasattr(layer_extraction,method):
        func = getattr(layer_extraction, method)
        if callable(func):
            layers = func(self.X,final_score,core_fraction,layer_extraction_params)

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
        self.layers_=None
        self.labels_=None

        self.set_labels=None
        self.set_layers=None


    # Layer extraction methods: {'quantile','partitioned_majority','majority','stable_rank'}
    def extract_layers(self, method='partitioned_majority', core_fraction=0.15, layer_extraction_params=None):

        layers_= _extract_layers(self, method=method, core_fraction=core_fraction,
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

