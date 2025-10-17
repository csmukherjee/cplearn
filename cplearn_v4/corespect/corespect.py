from dataclasses import asdict
from .config import (
    FlowRankConfig, StableCoreConfig,
    FineGrainedConfig, PropagationConfig
)
from .fine_grained_core import fine_grained_core
from .ranking import FlowRank
from .stable_core import stable_core

from ..utils.gen_utils import get_kNN

class CorespectModel:
    """
    CoreSpect model orchestrator.

    Stages:
        1. flowrank()              → node ranking and density estimation
        2. stable_core()           → stable core extraction
        3. fine_grained_core()     → optional refinement
        4. propagation_from_core() → diffusion or label spreading
    """

    def __init__(self, X,
                 flowrank_cfg=None,
                 stable_cfg=None,
                 fine_cfg=None,
                 prop_cfg=None):


        self.X = X
        self.flowrank_cfg = flowrank_cfg or FlowRankConfig()
        self.stable_cfg = stable_cfg or StableCoreConfig()
        self.fine_cfg = fine_cfg or FineGrainedConfig()
        self.prop_cfg = prop_cfg or PropagationConfig()

        self.knn_list = None
        self.knn_dist = None

        self.flowrank_score_ = None
        self.layers_ = None
        self.labels_ = None
        self.propagation_ = None

    def build_graph(self):
        """Construct adjacency list based on the largest neighborhood size."""
        q = self.flowrank_cfg.q
        r = self.flowrank_cfg.r
        ng = self.stable_cfg.ng_num
        k = max(q, r, ng)

        knn_list,knn_dist = get_kNN(self.X,k)

        self.knn_list = knn_list
        self.knn_dist = knn_dist

        return self

    def flowrank(self):
        self.flowrank_score_ = FlowRank(self.knn_list, **asdict(self.flowrank_cfg))
        return self

    def stable_core(self):
        self.layers_ = stable_core(self.X,self.knn_list, self.flowrank_score_, **asdict(self.stable_cfg))
        return self


    def fine_grained_core(self):
        self.layers_ =fine_grained_core(self.X,self.knn_list,self.layers_, **asdict(self.fine_cfg),**asdict(self.flowrank_cfg))

        return self


    # def propagation_from_core(self):
    #     self.propagation_ = propagate_from_core(
    #         self, **asdict(self.prop_cfg)
    #     )
    #     return self

    def run(self, fine_grained=False, propagate=True):
        self.build_graph()
        self.flowrank()
        self.stable_core()

        if fine_grained:

            print("Finding Fine Grained Core",flush=True)
            self.fine_grained_core()

        #self.cluster_core()

        # if propagate:
        #     self.propagation_from_core()

        return self

