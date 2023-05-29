#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:45:25 2023

@author: pasquale
"""

import heapq
import mcpg
import networkx as nx
import itertools
import numpy as np
import pandas as pd
import rbo
import seaborn as sns
import scipy
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
import utils


datasets =  ['./data/facebook-connected.csv', './data/git-connected.csv']
names = ['FB', 'Git']


def get_sparse_graph(graph):
    """
    Returns a sparse adjacency matrix in CSR format
    :param graph: undirected NetworkX graph
    :return: Scipy sparse adjacency matrix
    """

    return nx.to_scipy_sparse_array(graph, format='csr', dtype=float, nodelist=graph.nodes)

def get_node_ns(graph, k=3):
    """
    Get k nodes to attack based on the Netshield algorithm :cite`tong2010vulnerability`.
    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack
    :return: a list of nodes to attack
    """

    if not scipy.sparse.issparse(graph):
        sparse_graph = get_sparse_graph(graph)
    else:
        sparse_graph = graph

    lam, u = eigsh(sparse_graph, k=1, which='LA')
    lam = lam[0]

    u = np.abs(np.real(u).flatten())
    v = (2 * lam * np.ones(len(u))) * np.power(u, 2)

    nodes = []
    for i in range(k):
        B = sparse_graph[:, nodes]
        b = B * u[nodes]

        score = v - 2 * b * u
        score[nodes] = -1

        nodes.append(np.argmax(score))

    return nodes


def get_node_pr(graph, k=3):
    """
    Get k nodes to attack based on top PageRank entries :cite`page1999pagerank`.
    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack
    :return: a list of nodes to attack
    """

    centrality = nx.pagerank(graph, alpha=0.85)
    nodes = heapq.nlargest(k, centrality, key=centrality.get)

    return nodes


def get_node_eig(graph, k=3):
    """
    Get k nodes to attack based on top eigenvector centrality entries
    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack
    :return: a list of nodes to attack
    """

    centrality = nx.eigenvector_centrality(graph, tol=1E-3, max_iter=500)
    nodes = heapq.nlargest(k, centrality, key=centrality.get)
   
    return nodes


def get_node_id(graph,  k=3):
    """
    Get k nodes to attack based on Initial Degree (ID) Removal :cite:`beygelzimer2005improving`.
    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack
    :return: a list of nodes to attack
    """

    centrality = dict(graph.degree())
    nodes = heapq.nlargest(k, centrality, key=centrality.get)
  
    return nodes

def get_node_mcpg(graph,  k=3):
    """
    Get k nodes to attack based on Initial Degree (ID) Removal :cite:`beygelzimer2005improving`.
    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack
    :return: a list of nodes to attack
    """
    centrality = utils.potential_gain_standard(graph)
   
    
    nodes = heapq.nlargest(k, centrality, key=centrality.get)
   
    return nodes

def get_node_closeness(graph, k = 3):
    centrality = nx.closeness_centrality(graph)
    nodes = heapq.nlargest(k, centrality, key=centrality.get)
    return nodes
  

#graph = nx.fast_gnp_random_graph(1000, 0.1)
def compute_rbo(data_path, data_name):
    graph = nx.read_edgelist(data_path)
    print(graph.number_of_nodes())
    degree = get_node_id(graph, 100)
    eig = get_node_eig(graph, 100)
    clo = get_node_closeness(graph, 100)
    spg = get_node_mcpg(graph, 100)

    centrality_ranking = {'Deg': degree, 'Eig': eig,  'Clo': clo, 'SPG': spg}

    for k in [10, 20, 50, 100]:
        print('\centering')
        print('\caption{Correlation between centrality metrics in ' + data_name +' ($k =' +str(k)+'$)}')
        print('\label{table:'+str(k)+'-dataset:'+data_name+'}')
        print('\begin{tabular}{ll}')
        print('\toprule')
        print(' {\em Compared Methods}  & {\em RBO} ' + 'capo')
        print('\midrule')
        for (a, b) in itertools.combinations(list(centrality_ranking.keys()), 2):
            centrality_first = centrality_ranking[a][:k-1]
            centrality_second = centrality_ranking[b][:k-1]
            rbo_score = rbo.RankingSimilarity(centrality_first, centrality_second).rbo()
            print(a+'-'+b+ ' & '+ format(rbo_score, '.3f')+ ' capo')
            
        print('\bottomrule')
        print('\end{tabular}')
      
        print(10*'*')

# datasets =  ['./data/facebook-connesso.csv', './data/git-connesso.csv']
# names = ['FB', 'Git']

datasets =  ['./data/git-connesso.csv']
names = ['Git']



for i in range(2):
    #g = nx.read_edgelist(datasets[i])
    
    compute_rbo(datasets[i], names[i])
    print(20*'*')
    

    






