# %%
import os
import copy
import numpy as np
import networkx as nx
import dsd
from super_greedy_pp import *
from init_graph import *
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


def main():

    parser = argparse.ArgumentParser(description='General Tester')
    parser.add_argument('--formulation', type=int, default=1, metavar='R', help='Desired Formulation (1 for FADSG-I - 2 for FADSG-II)')
    parser.add_argument('--dataset-name', type=str, default='karate', metavar='S', help='Desired Dataset Name')
    parser.add_argument('--protected-class', type=float, default=1.0, metavar='R', help='Protected Class Label')
    
    args = parser.parse_args()

    formulation = copy.deepcopy(args.formulation)
    dataset_name = copy.deepcopy(args.dataset_name)
    protected_class = copy.deepcopy(args.protected_class)

    # %%
    G, protected_nodes, lam_max = init_graph(dataset_name, formulation, protected_class)

    n = G.number_of_nodes()
    print('Number of Nodes:', n)
    m = G.number_of_edges()
    print('Number of Edges:', m)

    # %%

    lam_vec=np.linspace(0.0, lam_max, num=100)

    # T=3
    # T=4
    T=5
    density_vec = []
    num_of_nodes = []
    num_of_protected_vec = []
    protected_portion_in_sub_vec = []
    protected_portion_in_prot_vec = []
    PoF_vec = []
    LB_UB_vec = []
    fairness_vec = []
    r_vec = []
    # Psi_vec = []

    print('MY super-greedy++ method')
    i=0
    for lam in tqdm(lam_vec):
        i=i+1
        super_greedy_pp_R = super_greedy_pp(G, T, protected_nodes, lam=lam, weight=None, formulation=formulation)
        super_greedy_pp_R_nodes = list(super_greedy_pp_R[0].nodes())
        super_greedy_pp_R_nodes.sort()
        if lam==0:
            density_0 =  utils.compute_density(super_greedy_pp_R[0])
        num_of_nodes.append(len(super_greedy_pp_R_nodes))
        
        density =  utils.compute_density(super_greedy_pp_R[0])
        density_vec.append(density)
        
        PoF = 1 - density/density_0
        PoF_vec.append(PoF)

        LB_UB = super_greedy_pp_R[3]
        LB_UB_vec.append(LB_UB)
        
        fairness = utils.find_num_common(super_greedy_pp_R_nodes, protected_nodes)
        fairness_vec.append(fairness)

        induced_protected_nodes = super_greedy_pp_R[2]
        r = utils.compute_r(super_greedy_pp_R[0], protected_nodes, induced_protected_nodes, formulation=formulation)
        r_vec.append(r)

        num_of_protected = len(induced_protected_nodes)
        num_of_protected_vec.append(num_of_protected)
        protected_portion_in_sub = num_of_protected/len(super_greedy_pp_R[0])
        protected_portion_in_sub_vec.append(protected_portion_in_sub)
        protected_portion_in_prot = num_of_protected/len(protected_nodes)
        protected_portion_in_prot_vec.append(protected_portion_in_prot)

    print('.done.')


    # %%
    # Save Variables in Dictionary
    variables_dict = {
        'lam': lam_vec,
        'induced': super_greedy_pp_R,
        'num_of_nodes': num_of_nodes,
        'density': density_vec,
        'num_of_protected': num_of_protected_vec,
        'protected_portion_in_sub': protected_portion_in_sub_vec,
        'protected_portion_in_prot': protected_portion_in_prot_vec,
        'PoF': PoF_vec,
        'fairness': fairness_vec,
        'r': r_vec,
        'LB_UB': LB_UB_vec
    }

    if "amazon" in dataset_name:
        save_folder = 'logs/'+'amazon'
        save_path = save_folder+'/'+dataset_name+'_'+str(formulation)+'_log.npy'
    elif "lastfm" in dataset_name:
        save_folder = 'logs/'+dataset_name
        save_path = save_folder+'/'+dataset_name+'_'+str(formulation)+'_'+str(protected_class)+'_log.npy'
    else:
        save_folder = 'logs/'+dataset_name
        save_path = save_folder+'/'+dataset_name+'_'+str(formulation)+'_log.npy'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(save_path, variables_dict)


if __name__ == '__main__':
    main()