#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 16:56:24 2022

@author: pasquale
"""
from collections import Counter
import numpy as np
import networkx as nx

def compute_mean(pg_dict):
    return {k : np.mean(np.array(pg_dict[k])) for k in pg_dict.keys()}

def compute_sum(pg_dict):
    return {k : np.sum(np.array(pg_dict[k])) for k in pg_dict.keys()}


# def potential_gain_standard(graph):
#     #kc_scores = nx.katz_centrality(graph, max_iter = 1000, tol = 1e-2)
#     #kc_scores = nx.katz_centrality(graph, max_iter = 50, tol = 1e-1)
#     kc_scores = nx.katz_centrality_numpy(graph)
#     deg_scores = nx.degree(graph)
#     pg_scores = {k: deg_scores[k]*kc_scores[k] for k in graph.nodes()}
#     return pg_scores

def potential_gain_standard(graph):
    deg_scores = dict(graph.degree())
    eig_scores = nx.eigenvector_centrality_numpy(graph)
    return {k: deg_scores[k]*eig_scores[k] for k in deg_scores}

def top_entries(scores):
    sorted_scores = Counter(scores)
    sorted_scores = sorted_scores.most_common(5)
    return list(map(lambda x: x[0], sorted_scores))

def extract_values_to_array(dict_scores):
    '''
    Takes a dictionary of pairs (node_id_potential_gain) and
    returns an np.array storing the potential gains

    Parameters
    ----------
    dict_scores : TYPE Dictionary
        DESCRIPTION. A dictionary in which nodes correspond to node IDs and values to their potential gain

    Returns
    -------
    An np array of potential gain scores

    '''
    return np.array(list(dict_scores.values())).reshape(1, -1)