#Main class for visualization.

from .coremap_algo import anchored_cmap


class Coremap:
    def __init__(
            self,
            X=None,
            round_info=None,
            labels=None,
            mode='three_steps',
            q=15,
            global_init_embedding=None,
            dict_viz=None
    ):

        #Attribute of Coremap

        # Attributes of the CoreSPECT
        self.X=X
        self.round_info = round_info
        self.final_labels=labels
        self.mode=mode
        self.q=q
        self.global_init_embedding=global_init_embedding
        self.dict_viz=dict_viz


        #self.dict_viz=_anchored_map(round_info=round_info,final_labels=labels,X=None,mode=mode,q=q)


    def anchored_map(self):
        dict_viz = anchored_cmap(self,self.X, self.round_info, self.final_labels, mode=self.mode, q=self.q,global_init_embedding=self.global_init_embedding)
        self.dict_viz=dict_viz


        return dict_viz



