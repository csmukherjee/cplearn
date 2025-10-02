# cplearn

**cplearn** is a Python toolkit for unsupervised learning on data with underlying **core–periphery-like** structures.  
The package includes:

- **CoreSPECT** – identifies most-to-least separable layers in the data w.r.t clustering, along with a clustering of most of the data  
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
import numpy as np
from sklearn.mixture import GaussianMixture

from cplearn.corespect import Corespect
from cplearn.visualizer import Visualizer as viz

# Generate synthetic 10D Gaussian Mixture data
n_samples = 500
n_components = 3
dim = 10
rng = np.random.RandomState(42)

# Define means and covariances
means = np.array([
    np.zeros(dim),
    np.ones(dim) * 2,
    np.ones(dim) * -2
])
cov = np.eye(dim) * 1.5
covariances = np.stack([cov] * n_components)

# Setup GMM manually
gmm = GaussianMixture(n_components=n_components, covariance_type='full')
gmm.weights_ = np.ones(n_components) / n_components
gmm.means_ = means
gmm.covariances_ = covariances
gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))

# Sample data
X, _ = gmm.sample(n_samples)


# Initialize and run CoreSPECT
corespect = Corespect(X)

# 1. Find dense core (e.g. 15% of data)
core = corespect.find_core(core_fraction=0.15)

# 2. Cluster core using Louvain
cluster_core = corespect.cluster_core(
    core,
    cluster_algo="louvain",
    cluster_algo_params={"ng_num": 20, "resolution": 1}
)

# 3. Propagate cluster labels outward
propagated_data = corespect.propagate_labels(
    cluster_core,
    propagate_algo="adaptive_majority", #Alternatively use "CDNN" to use the algorithm described in the CoreSPECT paper.
    propagate_algo_params={"ng_num": 20}
)

# 4. Extract layers and labels
layers, labels_for_layer = propagated_data.get_layers_and_labels()

# 5. Visualize (Visualizer internally calls CoreMAP for embedding)
mode_choice='three_steps' #use this with adaptive_majority
#You can use mode_choice='layerwise' to see a more in-depth layer-by-layer visualization
#Use layerwise if propagate_algo is "CDNN"
fig = viz(corespect,mode=mode_choice).fig
fig.show()   # or fig.write_html("corespect_viz.html")
```

---

## References

If you use this package in your research, please cite:

- **CoreSPECT**  
  Chandra Sekhar Mukherjee, Joonyoung Bae, and Jiapeng Zhang.  
  *CoreSPECT: Enhancing Clustering Algorithms via an Interplay of Density and Geometry.*
  *link: https://arxiv.org/abs/2507.08243 *

- *Recursive and adaptive majority propagation* – papers coming soon  
- *CoreMAP* – paper coming soon


### Other related work

- **Balanced Ranking**  
  Chandra Sekhar Mukherjee and Jiapeng Zhang.  
  *Balanced Ranking with Relative Centrality: A Multi-Core Periphery Perspective.*  
  ICLR 2025.

---

## License

This package is licensed under the GNU General Public License v3.0 (GPLv3).  
See the full license in the [`LICENSE`](./LICENSE) file.

---