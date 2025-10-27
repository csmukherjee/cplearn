# cplearn

[![PyPI version](https://badge.fury.io/py/cplearn.svg)](https://pypi.org/project/cplearn/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/cplearn.svg)](https://pypi.org/project/cplearn/)
[![Downloads](https://static.pepy.tech/badge/cplearn)](https://pepy.tech/project/cplearn)
[![arXiv](https://img.shields.io/badge/arXiv-2507.08243-b31b1b.svg)](https://arxiv.org/abs/2507.08243)


**cplearn** is a Python toolkit for unsupervised learning on data with underlying **core–periphery-like** structures.  
The package includes:

- **CoreSPECT** – identifies most-to-least separable layers in the data w.r.t clustering, along with a clustering.  
- **CoreMAP** –  Visualization w.r.t. underlying layered structure as derived by corespect using a novel anchor-based optimization.  
- **Visualizer** – interactive plots for visualizing core structure and subsequent layers  

---

## Installation

From PyPI:
```bash
pip install cplearn
```

---

## Quickstart

```python

#Generate mixture model based data for self-contained example.

import numpy as np

def generate_gmm_highdim(n=1000, d=10, gamma=1.0, seed=42):
    """
    Generate a 2-cluster Gaussian Mixture Model (GMM) in d dimensions.

    Parameters
    ----------
    n : int
        Total number of samples.
    d : int
        Dimensionality of the data (default 10).
    gamma : float
        Cluster separation factor. Lower gamma = harder to separate. [0.5=> hard]
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : (n, d) ndarray
        Generated data points.
    labels : (n,) ndarray
        True cluster labels (0 or 1).
    means : list of ndarray
        The two cluster means.
    """
    np.random.seed(seed)
    pi = [0.5, 0.5]  # equal mixture weights

    # Define means separated along the diagonal direction scaled by gamma
    base_sep = 1  # base distance between clusters
    mu1 = np.zeros(d)
    mu2 = np.ones(d) * base_sep * gamma

    # Slightly correlated covariance matrices
    A = np.eye(d)
    A += 0.2 * np.triu(np.ones((d, d)), 1)  # introduce mild correlation
    cov1 = np.dot(A, A.T) / d
    cov2 = cov1.copy()

    # Assign cluster labels
    labels = np.random.choice([0, 1], size=n, p=pi)

    # Sample from corresponding Gaussians
    X = np.zeros((n, d))
    X[labels == 0] = np.random.multivariate_normal(mu1, cov1, size=(labels == 0).sum())
    X[labels == 1] = np.random.multivariate_normal(mu2, cov2, size=(labels == 1).sum())

    return X, labels, [mu1, mu2]

#Generate data.
gamma=0.5
X, labels, means = generate_gmm_highdim(n=1000, d=10, gamma=gamma)

#---- The algorithm starts from here ----#


#Load CoreSPECT and configuration module
from cplearn.corespect import CorespectModel
from cplearn.corespect.config import CoreSpectConfig

#Initial parameters.
cfg = CoreSpectConfig(
    q=20,               #Determines neighborhood size for the underlying q-NN graph 
    r=10,               #Neighborhood radius parameter for ascending random walk with FlowRank
    core_frac=0.2,      #Fraction of points in the top-layer
    densify=False,      #Densifying different parts of the data to reduce fragmentation
    granularity=0.5,    #Higher granularity finds more local cores but can lead to missing out on weaker clusters.
    resolution=0.5      #Resolution for clustering with Leiden (more clustering methods will be added later)
).configure()

'''
For (q,r), two recommended choices are (40,20) and (20,10). 
(20,10) will lead to more fragmentation compared to (40,20).
'''

# Run **CoreSPECT**
model = CorespectModel(X, **cfg.unpack()).run(fine_grained=True,propagate=True)

'''
Main components:
model.layers_: Containts a list of lists. Each list consists of a subset of indices (between 0 and n-1, where n:= X.shape[0])
The first list corresponds to the indices that form the cores, the subsequent lists contain the outer layers.

model.labels_: n-sized integer array. 
    If propagate==False: Contains clustering label for the core (model.layers_[0]) indices, -1 in other places.
    If propagate==True:  Contains clustering label for all the points.

'''

#Visualizing the outcomes:

#Step 1: Generate UMAP skeleton.
import umap
reducer=umap.UMAP()
X_umap=reducer.fit_transform(X)


#Step 2: Initiate the **coremap** module.
from cplearn.coremap import Coremap
cmap=Coremap(model,global_umap=X_umap,fast_view=True)

'''
If fast_view= True, then we just use the UMAP skeleton, and then later show the visualization in a layer-wise manner.
If fast_view==False, we generate our own layer-wise visualization with the coremap algorithm.
'''


#Step 3: Layer-wise visualization (you can use your own labels instead of model.labels_)
from cplearn.coremap.vizualizer import visualize_coremap
fig=visualize_coremap(cmap,model.labels_, use_webgl=True)
fig.show()
```

---

## References

If you use this package in your research, please cite:

- **CoreSPECT**  
  Chandra Sekhar Mukherjee, Joonyoung Bae, and Jiapeng Zhang.  
  *CoreSPECT: Enhancing Clustering Algorithms via an Interplay of Density and Geometry.*
  *link: https://arxiv.org/abs/2507.08243 *

 
- *CoreMAP* – paper coming soon


### Other related work

- **Balanced Ranking**  
  Chandra Sekhar Mukherjee and Jiapeng Zhang.  
  *Balanced Ranking with Relative Centrality: A Multi-Core Periphery Perspective.*  
  ICLR 2025.

---

## License

This package is licensed under the BSD 3-Clause License.  
See the [LICENSE](./LICENSE) file for details.

---