from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics.cluster import adjusted_rand_score as ARI

from collections import Counter

from sklearn import metrics
import numpy as np

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def reverse_purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)


from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import _supervised



def ACC(label_ground,label_new):

    cm = confusion_matrix(label_ground,label_new)

    cm_s=-cm+np.max(cm)

    row_ind, col_ind = linear_sum_assignment(cm_s)
    acc= cm[row_ind, col_ind].sum()/np.sum(cm)

    return acc




def ACC_n(label, pred_label):
    value = _supervised.contingency_matrix(label, pred_label)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(label)

def cluster_acc(label_ground,label_new):

    label_ground_new = []
    label_new_new = []
    # remove -1 from label_ground and label_new
    for i in range(len(label_ground)):
        if label_new[i] != -1:
            label_ground_new.append(label_ground[i])
            label_new_new.append(label_new[i])
    label_ground = np.array(label_ground_new)
    label_new = np.array(label_new_new)

    #print(f"NMI={NMI(label_ground, label_new):.3f}, ARI={ARI(label_ground, label_new):.3f}, Purity={purity_score(label_ground, label_new):.3f}")
    return [NMI(label_ground, label_new),ARI(label_ground, label_new), purity_score(label_ground, label_new),reverse_purity_score(label_ground, label_new),ACC(label_ground,label_new)]

from sklearn.cluster import KMeans

def get_Kmeans(Y,true_k,return_cobjective=False,choose_min_obj=False,return_centroids=False):

    rounds=1

    if choose_min_obj:
        rounds=10

    min_obj_val = float('inf')
    for rr in range(rounds):
        kmeans = KMeans(n_clusters=true_k, n_init=1, max_iter=1000)
        kmeans.fit(Y)
        centroids = kmeans.cluster_centers_
        labels_km = kmeans.labels_

        obj_val = 0
        for ell in range(len(Y)):
            obj_val += np.linalg.norm(Y[ell] - centroids[labels_km[ell]]) ** 2

        if rr == 0 or obj_val < min_obj_val:
            min_obj_val = obj_val
            best_centroids = centroids
            best_labels_km = labels_km

    centroids = best_centroids
    labels_km = best_labels_km


    obj=0
    for i in range(len(labels_km)):
        obj += np.linalg.norm(Y[i] - centroids[labels_km[i]])**2

    if return_cobjective:
        return labels_km, obj

    if return_centroids:
        return labels_km, centroids

    return labels_km

def preserve_ratio(label,new_label):

    rel=1
    n1=len(label)
    n2=len(new_label)



    s1=Counter(label)
    s2=Counter(new_label)


    ratio=0

    for i in s1:
        if i in s2 :
            ratio=ratio+min((s2[i]/s1[i]),n2/(rel*n1))

    ratio=(ratio/n2)*(rel*n1)/len(set(label))

    return ratio