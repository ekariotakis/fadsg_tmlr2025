# import networkx as nx
import copy
import numpy as np
import utils
import sys


def super_greedy_pp(G, T, protected_nodes, lam, weight=None, formulation=1):
    protected_nodes_all = protected_nodes.copy()
    S_densest = G.copy()
    h_densest = utils.compute_cost_function(S_densest, protected_nodes_all, protected_nodes, lam, weight, formulation)
    protected_nodes_densest = protected_nodes.copy()
    
    n = G.number_of_nodes()
    loads = np.zeros((T, n))

    LB = h_densest
    UB = sys.maxsize

    LB_ = []
    UB_ = []
    LB_UB = []

    for i in range(1, T):
        LB_UB.append(LB/UB)
        LB_.append(LB)
        UB_.append(UB)
        
        S_i = G.copy()
        protected_nodes_i = set(protected_nodes)  # Assuming protected_nodes is a list, converting to set for O(1) lookups
        removed_nodes = set()

        for j in range(n):
            if S_i.number_of_nodes() == 1:
                break           

            v_star, prot_w_ = utils.find_min(S_i, loads[i,:], protected_nodes_all, list(protected_nodes_i), lam, list(removed_nodes), n, weight, formulation)

            protected_nodes_i.discard(v_star)
            
            v_star_deg = S_i.degree(v_star, weight)
            loads[i, v_star] = loads[i-1, v_star] + v_star_deg + prot_w_
            S_i.remove_node(v_star)
            removed_nodes.add(v_star)

            h_temp = utils.compute_cost_function(S_i, protected_nodes_all, protected_nodes_i, lam, weight, formulation)

            if h_temp > h_densest:
                LB = h_temp
                h_densest = h_temp
                S_densest = S_i.copy()
                protected_nodes_densest = protected_nodes_i.copy()

        UB = min(UB, max(loads[i, :]) / i)

    return S_densest, h_densest, protected_nodes_densest, LB_UB, i, LB_, UB_

