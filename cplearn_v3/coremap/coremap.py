#Main class for visualization.
import umap

from .viz_tools import anchored_map

class Coremap:
    def __init__(self,
        corespect_obj,
        global_umap=None
    ):

        self.core_obj=corespect_obj
        self.global_umap=global_umap
        self.X=corespect_obj.X
        self.G=corespect_obj.G
        self.X_umap=global_umap
        self.layers_=corespect_obj.layers_

        if self.global_umap is None:
            reducer = umap.UMAP(init='spectral')
            self.X_umap = reducer.fit_transform(self.X)

        label_dict=anchored_map(self)


        self.label_dict=label_dict








