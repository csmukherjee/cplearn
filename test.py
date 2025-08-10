from cplearn.corespect import Corespect

#Create test data from scikit learn
from sklearn.datasets import make_blobs
import numpy as np



def create_test_data(n_samples=1000, n_features=10, n_clusters=5, random_state=42):
    """
    Create synthetic test data using make_blobs and standardize it.
    """
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    
    return X, y

X, y= create_test_data()


"""
Test the Corespect class with synthetic data.
"""


# Initialize Corespect with the synthetic data
corespect = Corespect()
    
for algos in ['k_means', 'spectral_clustering',  'hdbscan']:
    # Fit the model
    if algos in ['louvain', 'hdbscan']:
        pred_labels = corespect.fit_predict(X = X, cluster_algo= algos)
    else:
        pred_labels = corespect.fit_predict(X = X, cluster_algo= algos, k = 5)
    
    # for name, val in vars(corespect).items():
    #     if name.startswith('_'):
    #         continue
    #     #print(f"{name}: {val}")
    #     print(f"{name}")



    #calculate NMI and ARI
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    nmi = normalized_mutual_info_score(y, pred_labels)
    ari = adjusted_rand_score(y, pred_labels)
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}")


