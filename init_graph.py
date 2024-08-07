# %%
import os
import copy
import numpy as np
import networkx as nx
# from dsd import *
import dsd
# from my_dsp import *
from super_greedy_pp import *
import utils
from datetime import datetime
import matplotlib.pyplot as plt 
import matplotlib
# %matplotlib inline 

from tqdm import tqdm

import random
import inspect

import csv

import argparse

def init_graph(dataset_name, formulation, protected_class=1.0):

    # %%
    if formulation==1:
        if dataset_name=="karate":
            lam_max = 1
        elif dataset_name=="polbooks":
            lam_max = 1.2
        elif dataset_name=="amazon_hpc":
            lam_max = 35
        elif dataset_name=="amazon_b":
            lam_max = 20
        elif dataset_name=="amazon_is":
            lam_max = 35
        elif dataset_name=="amazon_lb":
            lam_max = 0
        elif dataset_name=="amazon_op":
            lam_max = 50
        elif dataset_name=="amazon_ps":
            lam_max = 18

        elif dataset_name=="amazon_ah":
            lam_max = 40
        elif dataset_name=="amazon_acs":
            lam_max = 40
        elif dataset_name=="amazon_g":
            lam_max = 40
        elif dataset_name=="amazon_so":
            lam_max = 40
        elif dataset_name=="amazon_tmi":
            lam_max = 40

        elif dataset_name=="deezer":
            lam_max = 40

        elif dataset_name=="lastfm":
            lam_max = 40

        elif dataset_name=="github":
            lam_max = 100

        elif "twitch" in dataset_name:
            lam_max = 100

        elif dataset_name=="blogs":
            lam_max = 100

        elif dataset_name=="twitter":
            lam_max = 150

            
    elif formulation==2:
        if dataset_name=="karate":
            lam_max = 5
        elif dataset_name=="polbooks":
            lam_max = 5
        elif dataset_name=="amazon_hpc":
            lam_max = 70
        elif dataset_name=="amazon_b":
            lam_max = 12
        elif dataset_name=="amazon_is":
            lam_max = 150
        elif dataset_name=="amazon_lb":
            lam_max = 60
        elif dataset_name=="amazon_op":
            lam_max = 50
        elif dataset_name=="amazon_ps":
            lam_max = 50

        elif dataset_name=="amazon_ah":
            lam_max = 200
        elif dataset_name=="amazon_acs":
            lam_max = 50
        elif dataset_name=="amazon_g":
            lam_max = 140
        elif dataset_name=="amazon_so":
            lam_max = 50
        elif dataset_name=="amazon_tmi":
            lam_max = 50

        elif dataset_name=="deezer":
            lam_max = 125

        elif dataset_name=="lastfm":
            lam_max = 50

        elif dataset_name=="github":
            lam_max = 200

        elif "twitch" in dataset_name:
            lam_max = 200

        elif dataset_name=="blogs":
            lam_max = 100

        elif dataset_name=="twitter":
            lam_max = 200

    # %%
    if dataset_name == "karate":
        G = nx.karate_club_graph()
        
    elif dataset_name == "polbooks":
        graph_path = "datasets/pol_books/polbooks.gml"
        orig_G = nx.read_gml(graph_path)
        all_books = list(orig_G.nodes())
        neutral_books = [x for x,y in orig_G.nodes(data=True) if y['value']=='n']

        G_ = copy.deepcopy(orig_G)
        G_.remove_nodes_from(neutral_books)
        lib_cons_books = list(G_.nodes())
        G = copy.deepcopy(G_)

        mapping = dict(zip(G, range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)
        
    elif "amazon" in dataset_name:    
        
        if dataset_name == "amazon_hpc":
            graph_path = "datasets/amazon_dataset/All_Beauty__Health_&_Personal_Care.edges"
            color_path = "datasets/amazon_dataset/All_Beauty__Health_&_Personal_Care.colors"
        elif dataset_name == "amazon_b":
            graph_path = "datasets/amazon_dataset/All_Beauty__Baby.edges"
            color_path = "datasets/amazon_dataset/All_Beauty__Baby.colors"
        elif dataset_name == "amazon_is":    
            graph_path = "datasets/amazon_dataset/All_Beauty__Industrial_&_Scientific.edges"
            color_path = "datasets/amazon_dataset/All_Beauty__Industrial_&_Scientific.colors"
        elif dataset_name == "amazon_lb":    
            graph_path = "datasets/amazon_dataset/All_Beauty__Luxury_Beauty.edges"
            color_path = "datasets/amazon_dataset/All_Beauty__Luxury_Beauty.colors"
        elif dataset_name == "amazon_op":    
            graph_path = "datasets/amazon_dataset/All_Beauty__Office_Products.edges"
            color_path = "datasets/amazon_dataset/All_Beauty__Office_Products.colors"
        elif dataset_name == "amazon_ps":    
            graph_path = "datasets/amazon_dataset/All_Beauty__Pet_Supplies.edges"
            color_path = "datasets/amazon_dataset/All_Beauty__Pet_Supplies.colors"
        elif dataset_name == "amazon_ah": 
            graph_path = "datasets/amazon_dataset/All_Beauty__Amazon_Home.edges"
            color_path = "datasets/amazon_dataset/All_Beauty__Amazon_Home.colors"     
        elif dataset_name == "amazon_acs": 
            graph_path = "datasets/amazon_dataset/All_Beauty__Arts,_Crafts_&_Sewing.edges"
            color_path = "datasets/amazon_dataset/All_Beauty__Arts,_Crafts_&_Sewing.colors"
        elif dataset_name == "amazon_g": 
            graph_path = "datasets/amazon_dataset/All_Beauty__Grocery.edges"
            color_path = "datasets/amazon_dataset/All_Beauty__Grocery.colors"    
        elif dataset_name == "amazon_so": 
            graph_path = "datasets/amazon_dataset/All_Beauty__Sports_&_Outdoors.edges"
            color_path = "datasets/amazon_dataset/All_Beauty__Sports_&_Outdoors.colors" 
        elif dataset_name == "amazon_tmi": 
            graph_path = "datasets/amazon_dataset/All_Beauty__Tools_&_Home_Improvement.edges"
            color_path = "datasets/amazon_dataset/All_Beauty__Tools_&_Home_Improvement.colors"        
        
        
        G_ = nx.read_edgelist(graph_path, create_using=nx.Graph(), nodetype = int)
        
        colors = np.loadtxt(color_path)
        colors = colors.astype(int)
        
        for node_i in G_.nodes():
            color_value = colors[node_i,1]
            nx.set_node_attributes(G_, {node_i: color_value}, name="value")
        
        G = copy.deepcopy(G_)
        mapping = dict(zip(G, range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)
        
    elif "deezer" in dataset_name:    

        graph_path = "datasets/deezer_europe/deezer_europe_edges.csv"
        color_path = "datasets/deezer_europe/deezer_europe_target.csv"

        Data = open(graph_path, "r")
        next(Data, None)  # skip the first line in the input file
        G_ = nx.parse_edgelist(Data, delimiter=',', create_using=nx.Graph(),
                              nodetype=int, data=(('weight', float),))

        color_file = open(color_path, "r")
        next(color_file, None)
        data = list(csv.reader(color_file, delimiter=","))
        color_file.close()
        colors = np.array(data).astype(int)

        for node_i in G_.nodes():
            color_value = colors[node_i,1]
            nx.set_node_attributes(G_, {node_i: color_value}, name="value")

        G = copy.deepcopy(G_)
        mapping = dict(zip(G, range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)    
    
    elif "lastfm" in dataset_name:    

        graph_path = "datasets/lastfm_asia/lastfm_asia_edges.csv"
        color_path = "datasets/lastfm_asia/lastfm_asia_target.csv"

        Data = open(graph_path, "r")
        next(Data, None)  # skip the first line in the input file
        G_ = nx.parse_edgelist(Data, delimiter=',', create_using=nx.Graph(),
                            nodetype=int, data=(('weight', float),))

        color_file = open(color_path, "r")
        next(color_file, None)
        data = list(csv.reader(color_file, delimiter=","))
        color_file.close()
        colors = np.array(data).astype(int)

        for node_i in G_.nodes():
            color_value = colors[node_i,1]
            nx.set_node_attributes(G_, {node_i: color_value}, name="value")

        G = copy.deepcopy(G_)
        mapping = dict(zip(G, range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)     

    elif "github" in dataset_name:    

        graph_path = "datasets/git_web_ml/musae_git_edges.csv"
        color_path = "datasets/git_web_ml/musae_git_target.csv"

        Data = open(graph_path, "r")
        next(Data, None)  # skip the first line in the input file
        G_ = nx.parse_edgelist(Data, delimiter=',', create_using=nx.Graph(),
                            nodetype=int, data=(('weight', float),))

        color_file = open(color_path, "r")
        next(color_file, None)
        data = list(csv.reader(color_file, delimiter=","))
        color_file.close()
        colors_ = np.array(data)
        colors = np.delete(colors_, 1, 1)
        colors = colors.astype(int)

        for node_i in G_.nodes():
            color_value = colors[node_i,1]
            nx.set_node_attributes(G_, {node_i: color_value}, name="value")

        G = copy.deepcopy(G_)
        mapping = dict(zip(G, range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)
        
    elif "twitch" in dataset_name:    
        if "DE" in dataset_name:
            graph_path = "datasets/twitch/DE/musae_DE_edges.csv"
            color_path = "datasets/twitch/DE/musae_DE_target.csv"
        elif "ENGB" in dataset_name:
            graph_path = "datasets/twitch/ENGB/musae_ENGB_edges.csv"
            color_path = "datasets/twitch/ENGB/musae_ENGB_target.csv"
        elif "ES" in dataset_name:
            graph_path = "datasets/twitch/ES/musae_ES_edges.csv"
            color_path = "datasets/twitch/ES/musae_ES_target.csv"
        elif "FR" in dataset_name:
            graph_path = "datasets/twitch/FR/musae_FR_edges.csv"
            color_path = "datasets/twitch/FR/musae_FR_target.csv"
        elif "PTBR" in dataset_name:
            graph_path = "datasets/twitch/PTBR/musae_PTBR_edges.csv"
            color_path = "datasets/twitch/PTBR/musae_PTBR_target.csv"
        elif "RU" in dataset_name:
            graph_path = "datasets/twitch/RU/musae_RU_edges.csv"
            color_path = "datasets/twitch/RU/musae_RU_target.csv"


        Data = open(graph_path, "r")
        next(Data, None)  # skip the first line in the input file
        G_ = nx.parse_edgelist(Data, delimiter=',', create_using=nx.Graph(),
                            nodetype=int, data=(('weight', float),))

        color_file = open(color_path, "r")
        next(color_file, None)
        data = list(csv.reader(color_file, delimiter=","))
        color_file.close()
        # colors = np.array(data).astype(int)
        colors = np.array(data)

        for node_i in G_.nodes():
            color_value = colors[node_i,2] # mature
            nx.set_node_attributes(G_, {node_i: color_value}, name="value")

        G = copy.deepcopy(G_)
        mapping = dict(zip(G, range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)
    
    elif "blogs" in dataset_name:    

        graph_path = "datasets/blogs/out_graph.txt"
        color_path = "datasets/blogs/out_community.txt"

        G_ = nx.read_edgelist(graph_path, create_using=nx.Graph(), nodetype = int)
        
        colors = np.loadtxt(color_path)
        colors = colors.astype(int)
        
        for node_i in G_.nodes():
            color_value = colors[node_i,1]
            nx.set_node_attributes(G_, {node_i: color_value}, name="value")
        
        G = copy.deepcopy(G_)
        mapping = dict(zip(G, range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)
    
    elif "twitter" in dataset_name:    

        graph_path = "datasets/twitter/out_graph.txt"
        color_path = "datasets/twitter/out_community.txt"

        G_ = nx.read_edgelist(graph_path, create_using=nx.Graph(), nodetype = int)
        
        colors = np.loadtxt(color_path)
        colors = colors.astype(int)
        
        for node_i in G_.nodes():
            color_value = colors[node_i,1]
            nx.set_node_attributes(G_, {node_i: color_value}, name="value")
        
        G = copy.deepcopy(G_)
        mapping = dict(zip(G, range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)


    # %%
    n = G.number_of_nodes()
    print('Number of Nodes:', n)
    m = G.number_of_edges()
    print('Number of Edges:', m)

    # %%
    # * ***Color Graph's Protected Nodes***
    if dataset_name == "karate":
        ### Some different protected sets:
        yellow_set = [4, 5, 6, 10, 16]
        blue_set = [23, 24, 25, 27, 28, 31]
        green_set = [0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21]

        yellow_blue_set = yellow_set
        yellow_blue_set.extend(blue_set)
        yellow_blue_set.sort()
        
        protected_nodes = green_set    
        
    elif dataset_name == "polbooks":
        values_list = list(G.nodes(data='value'))
        protected_nodes = [item[0] for item in values_list if item[1] == 'l'] # protected = 'liberal'
               
    elif "amazon" in dataset_name or "deezer" in dataset_name:
        colors_list = list(G.nodes(data='value'))
        protected_nodes = [item[0] for item in colors_list if item[1] == 1] # protected = 1
    
    elif "lastfm" in dataset_name:
        colors_list = list(G.nodes(data='value'))
        if protected_class == 100:
            protected_nodes = [item[0] for item in colors_list if (item[1]==1) or (item[1]==2) or (item[1]==7) or (item[1]==9) or (item[1]==12) or (item[1]==13)]
            print('OK')
        else:
            protected_nodes = [item[0] for item in colors_list if item[1] == protected_class] # protected = protected_class
    
    elif "github" in dataset_name:
        colors_list = list(G.nodes(data='value'))
        protected_nodes = [item[0] for item in colors_list if item[1] == 1] # protected = 1
                    
    elif "twitch" in dataset_name:
        colors_list = list(G.nodes(data='value'))
        ## PTBR
        protected_nodes = [item[0] for item in colors_list if item[1] == 'True'] # 661 (for "mature" label)
        # protected_nodes = [item[0] for item in colors_list if item[1] == 'True'] # 279 (for "partner" label)
       
    elif "blogs" in dataset_name:
        colors_list = list(G.nodes(data='value'))
        protected_nodes = [item[0] for item in colors_list if item[1] == 0] # protected = 0 

    elif "twitter" in dataset_name:
        colors_list = list(G.nodes(data='value'))
        protected_nodes = [item[0] for item in colors_list if item[1] == 1] # protected = 0 

    protected_nodes.sort()    
    blue_nodes, red_nodes = utils.color_protected(G, protected_nodes) # blue = protected

    protected_density = utils.compute_density(protected_nodes, G)
        
   
    return G, protected_nodes, lam_max