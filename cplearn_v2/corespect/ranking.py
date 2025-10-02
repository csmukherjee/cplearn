import numpy as np
import networkx as nx
from numba import njit
from numba.typed import List

from ..utils.gen_utils import get_kNN

def init_random_walk(G,init_walk_len=10):
    n = G.number_of_nodes()
    v_cover=np.ones((n))

    hmap={}
    t=0
    for u in G.nodes():
        hmap[u]=t
        t+=1


    in_list = [[] for _ in range(n)]
    degs=np.zeros((n))
    for u in G.nodes():
        degs[hmap[u]]=G.out_degree(u)



    deg_list=[]
    for u in G.nodes():
        x=[]
        for v in G.predecessors(u):
            wt_uv=G.edges[v,u]['weight']
            x.append(wt_uv/degs[hmap[v]])
            in_list[hmap[u]].append(hmap[v])
        deg_list.append(np.array(x))


    #Now change v_cover:
    for ell in range(init_walk_len):

        v_cover_n=np.zeros((n))
        for i in range(n):
            v_cover_n[i]=sum(deg_list[i]*v_cover[in_list[i]])

        v_cover=v_cover_n.copy()

    init_score={}

    #Add init_score back to the graph
    for u in G.nodes():
        init_score[u]=np.float32(v_cover[hmap[u]])

    return init_score


@njit
def ascending_walk(ng_list,init_score,times):



    n0=len(init_score)
    v_cover=np.zeros((n0))

    walker=[]
    walker_ng_list=[]

    for ell in range(n0):
        x=[]
        for v in ng_list[ell]:
            if init_score[v]>init_score[ell]:
                x.append(v)


        walker.append(x)
        walker_ng_list.append(len(x))



    v_cover_n=np.zeros((n0))

    for rounds in range(times):

        #print("rounds=",rounds)

        v_cover=np.zeros((n0))

        for u in range(n0):
            v=u
            while walker_ng_list[v]>0:
                v= walker[v][np.random.randint(0, walker_ng_list[v])]

            v_cover[u]=init_score[v]

        v_cover_n=v_cover_n+v_cover


    v_cover_n=v_cover_n/times

    return v_cover_n



def FLOW_rank_optimized(X,init_score,r):

    times=200

    knn_list, _=get_kNN(X, r)
    G=nx.DiGraph()
    for i in range(len(knn_list)):
        for j in knn_list[i]:
            G.add_edge(i,j,weight=1)

    n=X.shape[0]



    ng_list=[]

    #Preparing for numba. ng_list is a n*r matrix.
    for ell in range(n):
      x=[]
      for v in G.neighbors(ell):
        x.append(int(v))

      ng_list.append(List(x))

    init_score_numba=np.zeros((n))
    for u in init_score:
        init_score_numba[u]=init_score[u]

    v_cover_n=ascending_walk(ng_list,init_score_numba,times)

    final_score={}
    for u in G.nodes():
        final_score[u]=init_score[u]/(v_cover_n[u]+0.00001)




    return final_score


def FlowRank(X,ranking_algo_params):
    allowed_params = ['q', 'r']
    for key in ranking_algo_params.keys():
        if key not in allowed_params:
            raise ValueError(f"Unwanted parameter found: {key}")

    q = ranking_algo_params.get('q', 40)
    r = ranking_algo_params.get('r', 20)

    #Get initial density estimation.

    knn_list, _=get_kNN(X, q)
    G=nx.DiGraph()
    for i in range(len(knn_list)):
        for j in knn_list[i]:
            G.add_edge(i,j,weight=1)


    init_score=init_random_walk(G)

    final_score=FLOW_rank_optimized(X,init_score,r)


    return final_score

#FlowRank based on number of hops
@njit
def ascending_walk_count_hops(ng_list,init_score,times):



    n0=len(init_score)
    v_cover=np.zeros((n0))

    walker=[]
    walker_ng_list=[]

    for ell in range(n0):
        x=[]
        for v in ng_list[ell]:
            if init_score[v]>init_score[ell]:
                x.append(v)


        walker.append(x)
        walker_ng_list.append(len(x))



    v_cover_n=np.zeros((n0))

    for rounds in range(times):

        #print("rounds=",rounds)

        v_cover=np.zeros((n0))

        for u in range(n0):
            c=0
            v=u
            while walker_ng_list[v]>0:
                v= walker[v][np.random.randint(0, walker_ng_list[v])]
                c+=1

            v_cover[u]=c

        v_cover_n=v_cover_n+v_cover


    v_cover_n=v_cover_n/times

    return v_cover_n


def FLOW_rank_optimized_hops(X,init_score,r):

    times=200

    knn_list, _=get_kNN(X, r)
    G=nx.DiGraph()
    for i in range(len(knn_list)):
        for j in knn_list[i]:
            G.add_edge(i,j,weight=1)

    n=X.shape[0]



    ng_list=[]

    #Preparing for numba. ng_list is a n*r matrix.
    for ell in range(n):
      x=[]
      for v in G.neighbors(ell):
        x.append(int(v))

      ng_list.append(List(x))

    init_score_numba=np.zeros((n))
    for u in init_score:
        init_score_numba[u]=init_score[u]

    v_cover_n=ascending_walk_count_hops(ng_list,init_score_numba,times)

    final_score={}
    for u in G.nodes():
        final_score[u]=-v_cover_n[u]




    return final_score



def FlowRank_count_hops(X,ranking_algo_params):
    allowed_params = ['q', 'r']
    for key in ranking_algo_params.keys():
        if key not in allowed_params:
            raise ValueError(f"Unwanted parameter found: {key}")

    q = ranking_algo_params.get('q', 40)
    r = ranking_algo_params.get('r', 20)

    #Get initial density estimation.

    knn_list, _=get_kNN(X, q)
    G=nx.DiGraph()
    for i in range(len(knn_list)):
        for j in knn_list[i]:
            G.add_edge(i,j,weight=1)


    init_score=init_random_walk(G)

    final_score=FLOW_rank_optimized_hops(X,init_score,r)


    return final_score

import numpy as np
from collections import defaultdict

#Standard rankings: Degree and PageRank
def PageRank(X,ranking_algo_params):
    allowed_params = ['q', 'r']
    for key in ranking_algo_params.keys():
        if key not in allowed_params:
            raise ValueError(f"Unwanted parameter found: {key}")

    q = ranking_algo_params.get('q', 40)
    r = ranking_algo_params.get('r', 20)

    alpha=0.85

    knn_list, _=get_kNN(X, q)
    G=nx.DiGraph()
    for i in range(len(knn_list)):
        for j in knn_list[i]:
            G.add_edge(i,j,weight=1)

    pagerank_scores = nx.pagerank(G,alpha=alpha)

    return pagerank_scores

