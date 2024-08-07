from re import X
import networkx as nx
import copy
import numpy as np
from super_greedy_pp import *
import colorsys
import matplotlib
from datetime import datetime


def color_protected(G, protected_nodes, print_=False):
    protected_nodes.sort()
    
    for node_i in range(len(G.nodes())):
        if node_i in protected_nodes:
            nx.set_node_attributes(G, {node_i: 'blue'}, name="color")
        else:
            nx.set_node_attributes(G, {node_i: 'red'}, name="color")

    blue_nodes = [x for x,y in G.nodes(data=True) if y["color"]=="blue"]
    if print_==True:
        print("Protected (Blue) Nodes: ", blue_nodes)
    red_nodes = [x for x,y in G.nodes(data=True) if y["color"]=="red"]
    if print_==True:
        print("Un-Protected (Red) Nodes: ", red_nodes)

    return blue_nodes, red_nodes

def find_complement(omega_vec, x_vec):
    x_set = set(x_vec)
    omega_set = set(omega_vec)
    return list(omega_set - x_set)

def find_common(x_vec, y_vec):
    x_set = set(x_vec)
    y_set = set(y_vec)

    # Find the common elements by taking the intersection of the sets
    common_elements = x_set.intersection(y_set)
    
    return common_elements

def find_num_common(x_vec, y_vec):
    # Find the common elements by taking the intersection of the sets
    common_elements = find_common(x_vec, y_vec)

    # Get the number of common elements
    num_common = len(common_elements)
    
    return num_common

def find_protected_portion(x_vec, protected_vec):
    num_common = find_num_common(x_vec, protected_vec)
    return num_common / len(x_vec)

def compute_r(S, protected_nodes_all, protected_nodes, formulation=1):
    """
    compute r(S),
    FADSG-I: |S^S_p|/|S|
    FADSG-II: dist(S,S_p)/|S|
    """

    if formulation==1:
        num_nodes = S.number_of_nodes()
        num_protected = len(protected_nodes)
        r = num_protected/num_nodes
    elif formulation==2:
        num_nodes = S.number_of_nodes()
        num_protected_all = len(protected_nodes_all)
        num_protected = len(protected_nodes)
        r = -(num_nodes+num_protected_all-2*num_protected)/num_nodes

    return r  

def compute_f(S, protected_nodes_all, protected_nodes, lam, weight=None, formulation=1):
    """
    compute F(S_i) 
    """

    if formulation==1:
        num_edges = S.size(weight)
        num_protected = len(protected_nodes)
        f = num_edges + lam*num_protected
    elif formulation==2:
        num_edges = S.size(weight)
        num_nodes = S.number_of_nodes()
        num_protected_all = len(protected_nodes_all)
        num_protected = len(protected_nodes)
        f = num_edges - lam*(num_nodes+num_protected_all-2*num_protected)

    return f

def compute_cost_function(S, protected_nodes_all, protected_nodes_i, lam, weight, formulation=1): 
    return compute_f(S, protected_nodes_all, protected_nodes_i, lam, weight, formulation) / S.number_of_nodes()

def compute_density(S, G=None, weight=None):
    if type(S) is not nx.classes.graph.Graph:
        S = G.subgraph(S)

    return S.size(weight) / S.number_of_nodes()
    


### FIND_MIN 

def find_min(S, loads, protected_nodes_all, protected_nodes, lam, removed_nodes, n, weight=None, formulation=1):
    """
    Find node 'v' with min: l+F(v|S-v)
    """
    
    f = np.zeros(n)
    prot_w = np.zeros(n)
    S_list = list(S.nodes())
    S_list.sort() ## this may be unnecessary

    # S_list : S
    # protected_nodes : S int Sp
    # protected_nodes_all : Sp
    S_p = list(find_common(S_list, protected_nodes)) # S int Sp
    S_np = list(find_complement(S_list, S_p)) # S int (S int Sp)'

    S_only_all = list(find_complement(protected_nodes_all, protected_nodes)) # Sp int (S int Sp)'
    S_not_all = list(find_complement(S_list, protected_nodes_all)) # S int Sp'

    S_number_of_nodes = S.number_of_nodes()
    len_protected_nodes = len(protected_nodes)
    
    if formulation==1:
        prot_w[S_p] = lam
    elif formulation==2:
        prot_w[protected_nodes] = lam
        # prot_w[S_only_all] = -2*lam
        prot_w[S_not_all] = -lam
    S_deg = np.array(list(S.degree(S_list, weight)))
    f[S_list] = S_deg[:,1] + prot_w[S_list]
    loads_f = loads + f

    ## Mask out removed nodes in order to get the correct min
    new_loads_f = np.array(loads_f)
    m = np.zeros(new_loads_f.size, dtype=bool)
    m[removed_nodes] = True
    masked_loads_f = np.ma.array(new_loads_f, mask=m)
    arg_min = np.argmin(masked_loads_f)
    v = arg_min
    
    return v, prot_w[v]

###########################
