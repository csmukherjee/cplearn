""" CoreSPECT Implementation"""

from . import ranking, cluster_core, propagate
from .cav import calculate_cav
import numpy as np
from ..utils.processing import extract_layers_labels



def _calc_rank(self):
    # If scores are not provided, rank using ranking_algo
    if self.scores is None:
        if hasattr(ranking, self.ranking_algo):
            func = getattr(ranking, self.ranking_algo)
            if callable(func):
                self.scores = func(self.X, self.ranking_algo_params)
            else:
                raise TypeError(f"{self.ranking_algo} is not callable")

        else:
            raise KeyError(f"{self.ranking_algo} is not a valid ranking algorithm")
        print(f"Obtained vertex ranking using {self.ranking_algo}")

    else:
        print("Ranking with passed scores")

    self.sorted_points = np.array(sorted(self.scores, key=self.scores.get, reverse=True)).astype(int)


    if self.core_fraction is None:
        raise KeyError("The Core fraction cannot be None")

    self.core_nodes = self.sorted_points[0:int(self.core_fraction * self.n)]

def _cluster_core_nodes(self):
    # Cluster the core_nodes using cluster_algo
    if hasattr(cluster_core, self.cluster_algo):
        func = getattr(cluster_core, self.cluster_algo)
        if callable(func):
            self.core_labels= func(self.X, core_nodes=self.core_nodes,
                                                                        cluster_algo_params=self.cluster_algo_params)
        else:
            raise TypeError(f"{self.cluster_algo} is not callable")

    else:
        raise KeyError(f"{self.cluster_algo} is not a valid clustering algorithm")


    print("Number of clusters found in the core:", len(set(self.core_labels)))


def _propagate_labels(self):
    self.cluster_assignment_vectors = calculate_cav(X_core=self.X[self.core_nodes], core_labels=self.core_labels,
                                                    cav=self.cav)
    
    # Start generating final labels
    self.final_labels = -1 * np.ones(self.n)
    self.final_labels[self.core_nodes] = self.core_labels
    self.final_labels = self.final_labels.astype(int)
    self.round_info=None

    # Label the rest (or some of the rest) of the points using the propagate_algo algorithm
    if hasattr(propagate, self.propagate_algo):
        func = getattr(propagate, self.propagate_algo)
        if callable(func):

            if self.propagate_algo=='CDNN':

               self.final_labels,self.round_info = func(self.X, self.sorted_points, self.core_nodes, self.final_labels,
                                        self.cluster_assignment_vectors, self.propagate_algo_params)

            elif self.propagate_algo =='recursive_majority' or self.propagate_algo=='adaptive_majority':

                self.final_labels,self.round_info = func(self.X, self.core_nodes, self.final_labels, self.propagate_algo_params)

            else:
                raise KeyError(f"{self.propagate_algo} is not a valid propagation algorithm now")

        else:
            raise TypeError(f"{self.propagate_algo} is not callable")

    else:
        raise KeyError(f"{self.propagate_algo} is not a valid propagation algorithm")

class Core: #Inherits all the functions including __init__
    def __init__(self, parent: "Corespect", scores, ranking_algo, core_fraction, ranking_algo_params):
        # attributes of Data
        self.X = parent.X # np array reference, not a copy 
        self.n = parent.n

        # attributes for children of this Core object (Clustered_Core)
        self.children = []

        # attributes for ranking algorithm
        self.ranking_algo_params = ranking_algo_params
        self.scores = scores
        self.ranking_algo = ranking_algo
        self.core_fraction = core_fraction

        # attributes updated in _calc_rank
        self.core_nodes = None
        self.sorted_points = None

        _calc_rank(self)

class Clustered_Core:
    def __init__(self, parent: "Core", cluster_algo, cluster_algo_params):
        # attributes of Data
        self.X = parent.X
        self.n = parent.n

        # attributes for children of this Clustered_Core object (Propagated_Data)
        self.children = []

        # attributes for clustering algorithm
        self.core_nodes = parent.core_nodes
        self.cluster_algo = cluster_algo
        self.cluster_algo_params = cluster_algo_params
        self.sorted_points = parent.sorted_points

        # attributes updated in _cluster_core_nodes
        self.core_labels = None

        _cluster_core_nodes(self)

class Propagated_Data:
    def __init__(self, parent: "Corespect", core_labels, cav, propagate_algo, propagate_algo_params):
        # attributes of Data
        self.X = parent.X
        self.n = parent.n

        # attributes for propagation algorithm
        if core_labels is None:
            self.core_labels = parent.core_labels
        self.core_nodes = parent.core_nodes
        self.cav = cav
        self.propagate_algo = propagate_algo
        self.propagate_algo_params = propagate_algo_params
        self.sorted_points = parent.sorted_points

        # attributes updated in _propagate_labels
        self.final_labels = None
        self.cluster_assignment_vectors = None
        self.round_info = None


        _propagate_labels(self)

    def get_layers_and_labels(self):

        if self.propagate_algo == 'CDNN':
            mode='layerwise'
        elif self.propagate_algo == 'recursive_majority':
            mode='three_steps'
        elif self.propagate_algo == 'adaptive_majority':
            mode='three_steps'
        else:
            raise ValueError(f"{self.propagate_algo} is not a valid propagation algorithm for now")

        layers,labels_for_layer=extract_layers_labels(self.round_info,self.final_labels,mode=mode)
        return layers,labels_for_layer




class Corespect:
    def __init__(
            self,
            X=None,
    ):
        # Attributes of the CoreSPECT (CDNN)
        self.X = X
        self.n = X.shape[0]

        # Attributes for children of corespect object (Core, Clustered_Core, Propagated_Data)
        self.children = []

        # Attributes for ranking
        self.ranking_algo = None
        self.core_fraction = None
        self.ranking_algo_params = {}
        self.scores = None

        # Attributes for core clustering
        self.cluster_assignment_vectors = None
        self.sorted_points = None
        self.core_labels = None
        self.core_nodes = None
        self.final_labels = None
        self.cluster_algo = None
        self.cluster_algo_params = {}
        
        # Attributes for propagation
        self.cav = None
        self.propagate_algo = None
        self.num_step = None # Todo: merge into propagate_algo_params
        self.propagate_algo_params = {}

    _calc_rank = _calc_rank
    _cluster_core_nodes = _cluster_core_nodes
    _propagate_labels = _propagate_labels

    def find_core(self, scores=None, ranking_algo='FlowRank', core_fraction=0.1, ranking_algo_params={}):
        core = Core(self, scores, ranking_algo, core_fraction, ranking_algo_params)
        self.children.append(core)  # Store the core object in children
        return core

    def cluster_core(self, core: Core, cluster_algo='k_means', cluster_algo_params={}):
        clustered_core = Clustered_Core(core, cluster_algo, cluster_algo_params)
        core.children.append(clustered_core)  # Store the clustered core object in children
        return clustered_core

    def propagate_labels(self, clustered_core: Clustered_Core, core_labels=None, cav='dist', propagate_algo='CDNN', propagate_algo_params={}):
        propagated_data = Propagated_Data(clustered_core, core_labels, cav, propagate_algo, propagate_algo_params)
        clustered_core.children.append(propagated_data)  # Store the propagated data object in children
        return propagated_data

    def fit_predict(self, X, ranking_algo='FlowRank', core_fraction=0.1,
                    cluster_algo='k_means',cav='dist',
                cluster_algo_params={}, propagate_algo='CDNN', propagate_algo_params={}):

        self.X = X
        self.n = X.shape[0]

        self.ranking_algo = ranking_algo
        self.core_fraction = core_fraction

        self.cluster_algo = cluster_algo
        self.cav = cav
        self.cluster_algo_params = cluster_algo_params


        self.propagate_algo = propagate_algo
        self.propagate_params = propagate_algo_params

        self._calc_rank()
        self._cluster_core_nodes()
        self._propagate_labels()

        return self.final_labels


