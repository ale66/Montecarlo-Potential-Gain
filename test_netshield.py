#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:16:16 2023

@author: pasquale
"""

import matplotlib.pyplot as plt
import NetShieldSolver as NSS
import numpy as np
import networkx as nx
import utils
import pandas as pd
import heapq
import Graph_Sampling

datasets =  ['./data/facebook-connected.csv', './data/git-connected.csv']

names = ['FB', 'Git']


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

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


def compare_with_nss_old(datapath, dataname, iterations):
    graph = nx.read_edgelist(datapath)
    #graph = nx.fast_gnp_random_graph(200, 0.1)
    print(dataname, graph.number_of_nodes())
    df_result = pd.DataFrame(columns = ['k', 'Deg', 'Eig', 'SPG'])
    for k in range(1, 21, 2):
        deg_scores  = np.zeros(iterations)
        eig_scores = np.zeros(iterations)
        spg_scores = np.zeros(iterations)
        for it in range(iterations):
            object_graph = Graph_Sampling.SRW_RWF_ISRW()             
            sampled_subgraph = object_graph.random_walk_induced_graph_sampling(graph, 2000)
            NSS_solver = NSS.NetShieldSolver(sampled_subgraph, seeds  = [1, 2], k = k)
            NSS_solver.run()
            
            degree = get_node_id(sampled_subgraph, k)
            eig = get_node_eig(sampled_subgraph, k)
            spg = get_node_mcpg(sampled_subgraph, k)
        
        
            deg_scores[it] = jaccard_similarity(degree,  NSS_solver.log['Blocked nodes'])
            eig_scores[it] = jaccard_similarity(eig,  NSS_solver.log['Blocked nodes'])
            spg_scores[it] = jaccard_similarity(spg,  NSS_solver.log['Blocked nodes'])
    
        df_result.loc[k] = [k, np.mean(deg_scores), np.mean(eig_scores), np.mean(spg_scores)]
        
        # deg_nss = jaccard_similarity(degree,  NSS_solver.log['Blocked nodes'])
        # eig_nss = jaccard_similarity(eig,  NSS_solver.log['Blocked nodes'])
        # spg_nss = jaccard_similarity(spg,  NSS_solver.log['Blocked nodes'])
    
    #print(k, np.mean(deg_scores), np.mean(eig_scores), np.mean(spg_scores))
    df_result.to_csv('./results/comparison-nss-'+dataname, index = False)
        

def compare_with_nss(dataname, iterations, p):
   
    print(dataname, p)
    
    df_result = pd.DataFrame(columns = ['k', 'Deg', 'Eig', 'SPG'])
    for k in range(1, 22, 2):
        deg_scores  = np.zeros(iterations)
        eig_scores = np.zeros(iterations)
        spg_scores = np.zeros(iterations)
        for it in range(iterations):
                       
            sampled_subgraph = nx.fast_gnp_random_graph(1200,p)
            NSS_solver = NSS.NetShieldSolver(sampled_subgraph, seeds  = [1, 2], k = k)
            NSS_solver.run()
            
            degree = get_node_id(sampled_subgraph, k)
            eig = get_node_eig(sampled_subgraph, k)
            spg = get_node_mcpg(sampled_subgraph, k)
        
        
            deg_scores[it] = jaccard_similarity(degree,  NSS_solver.log['Blocked nodes'])
            eig_scores[it] = jaccard_similarity(eig,  NSS_solver.log['Blocked nodes'])
            spg_scores[it] = jaccard_similarity(spg,  NSS_solver.log['Blocked nodes'])
    
        df_result.loc[k] = [k, np.mean(deg_scores), np.mean(eig_scores), np.mean(spg_scores)]
 
    print(df_result)
    print(dataname)
    df_result.to_csv(dataname, index = False)


def make_plot(inputfile, outputfile):
    #df = pd.read_csv('./results/comparison-nss-'+dataname)
    df = pd.read_csv(inputfile)
    print(df.head())
    linestyles = ['dotted', 'dashed', 'dashdot']
    markers = ['.', 'o', 'x']
    plt.clf()
    for i, col in enumerate(df.columns):
        if col != 'k':
            plt.plot(df['k'], df[col], label = col, linestyle = linestyles[i-1], marker = markers[i-1], markersize = 4, linewidth = 1.5)
    plt.xticks(range(1, 21, 2))
    plt.legend(loc = 'best')
    plt.grid()
    plt.xlabel(r'$k$', fontsize = 12)
    plt.ylabel('Jaccard Coefficient', fontsize = 12)
    plt.savefig(outputfile, dpi = 300)
    #plt.savefig('./figures/jaccard-nss-'+dataname+'.jpg', dpi = 300)

make_plot('Facebook-Friend', 'jaccard-facebook.jpg')
make_plot('git.csv', 'jaccard-git.jpg')
# compare_with_nss('Facebook-Friend', 70, 0.05)   
# compare_with_nss('GitHub', 70, 0.001)
# for d in datasets:
    # print(d)
    # graph = nx.read_edgelist(d)
    # print(nx.density(graph))


# for (n, d) in zip(names, datasets):
#     print(n, d)
#     compare_with_nss(d, 70)
#     #make_plot(n)


#df_result = pd.DataFrame(columns = ['Trial', 'Deg', 'Eig', 'SPG'])

# for k in [1, 3, 5, 10, 15]:
#     df_result = pd.DataFrame(columns = ['Trial', 'Deg', 'Eig', 'SPG'])
#     #graph = nx.read_edgelist(b)
#     for it in range(5):
#         graph = nx.fast_gnp_random_graph(100, 0.1)
#         NSS_solver = NSS.NetShieldSolver(graph, seeds  = [1, 2, 3, 4, 5], k = k)
#         NSS_solver.run()
        
#         degree = get_node_id(graph, k)
#         eig = get_node_eig(graph, k)
#         spg = get_node_mcpg(graph, k)
        
#         deg_nss = jaccard_similarity(degree,  NSS_solver.log['Blocked nodes'])
#         eig_nss = jaccard_similarity(eig,  NSS_solver.log['Blocked nodes'])
#         spg_nss = jaccard_similarity(spg,  NSS_solver.log['Blocked nodes'])
        
#         df_result.loc[it] = [it, deg_nss, eig_nss, spg_nss]
#     print(df_result)
    
    # print('Deg', degree, jaccard_similarity(degree,  NSS_solver.log['Blocked nodes']))
    # print('Eig', eig, jaccard_similarity(eig,  NSS_solver.log['Blocked nodes']))
    # print('SPG', spg, jaccard_similarity(spg,  NSS_solver.log['Blocked nodes']))
    
    # print('Deg', jaccard_similarity(degree,  NSS_solver.log['Blocked nodes']))
    # print('Eig', jaccard_similarity(eig,  NSS_solver.log['Blocked nodes']))
    # print('SPG', jaccard_similarity(spg,  NSS_solver.log['Blocked nodes']))

