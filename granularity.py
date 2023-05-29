#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:31:27 2023

@author: pasquale
"""


import heapq
import matplotlib.pyplot as plt
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

def normalize_spg(spg_scores):
    lowest = min(spg_scores.values())
    largest = max(spg_scores.values())
    delta = largest - lowest
    return {k: (spg_scores[k] - lowest)/delta for k in spg_scores}
    

def make_plot(input_dataset, output_fig):
    graph = nx.read_edgelist(input_dataset)
    #graph = nx.fast_gnp_random_graph(100, 0.1)
    centralities = dict()
    centralities['Deg'] = np.array(list(nx.degree_centrality(graph).values()))
    centralities['Eig'] = np.array(list(nx.eigenvector_centrality(graph, tol=1E-3, max_iter=500).values()))
    centralities['SPG'] = np.array(list(normalize_spg(utils.potential_gain_standard(graph)).values()))
    
    print(float(len(np.unique(centralities['Deg'])))/len(centralities['Deg']))
    
    for k in centralities.keys():
        print(k)
        n = float(len(np.unique(centralities[k])))
        d = len(centralities[k])
        print(n/d)
    
    # centralities['Deg'] = np.array(list(dict(graph.degree()).values()), dtype = float)
    # centralities['Eig'] = np.array(list(nx.eigenvector_centrality(graph, tol=1E-3, max_iter=500).values()), dtype = float)
    # centralities['SPG'] = np.array(list(utils.potential_gain_standard(graph).values()), dtype = float)
    
    #print(np.split(np.sort(centralities['Deg']), 10))
    
    #print(centralities)
    
    # plt.clf()
    # for centrality in centralities:
    #     counts, bins = np.histogram(centralities[centrality], 50)
    #     #print(counts)
    #     c = 100*(counts/np.sum(counts))
    #     print(c[:3])
    #     #sns.displot(centralities[centrality])
    #     plt.stairs(counts, bins, label = centrality)
    # #     sorted_data = np.sort(centralities[centrality])
    # #     plt.step(np.concatenate([sorted_data, sorted_data[[-1]]]), np.arange(sorted_data.size+1), label = centrality)
    
    # plt.yscale('log')
    # plt.legend(loc = 'best')
    # plt.ylabel('Number of unique elements', fontsize = 12)
    # plt.xlabel('Bins')
    # plt.savefig(output_fig, dpi = 300)


datasets =  ['./data/facebook-connected.csv', './data/git-connected.csv']

names = ['FB', 'Git']

for (a, b) in zip(names, datasets):
    print(b)
    make_plot(b, 'granularity-'+a+'.jpg')


#graph = nx.fast_gnp_random_graph(1000, 0.1)
# degrees = nx.degree_centrality(graph)
# print(min(degrees.values()), max(degrees.values()))
# eigs = nx.eigenvector_centrality(graph, tol=1E-3, max_iter=500)
# print(min(eigs.values()), max(eigs.values()))
# spg = utils.potential_gain_standard(graph)
# print(min(spg.values()), max(spg.values()))
# w = normalize_spg(spg)
# print(w)
#make_plot(graph)
